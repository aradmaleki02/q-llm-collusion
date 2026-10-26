from concurrent.futures import ThreadPoolExecutor
import json
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import torch

from collusion.api import Gateway
from collusion.config import Config
from collusion.convergence import advance_training
from collusion.engine import make_experiment, run_experiment
from collusion.parallel import initialize_parallel, run_parallel
from collusion.learner import Network
from collusion.storage import Paused, Store


POLICY = dict(authorization='Parallel test', minimum_games=2, return_window_games=1,
    return_relative_tolerance=1., policy_change_tolerance=1., q_drift_tolerance=1.,
    required_stable_checks=1, probe_source_games=1, probe_prefixes_per_game=2,
    extra_replay_updates_per_game=2, evaluation_reserve_usd=0.)
DEVICES = {'ordinary':'cpu','optimistic':'cpu'}


class ParallelTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.device_count()>=2,'Two CUDA GPUs unavailable')
    def test_both_cuda_devices_execute_their_own_networks(self):
        original = Network.forward
        seen = set()
        def forward(network,sequences):
            seen.add(str(next(network.parameters()).device))
            return original(network,sequences)
        with patch.object(Network,'forward',forward):
            result = run_parallel(self.store,'study',POLICY,
                                  {'ordinary':'cuda:0','optimistic':'cuda:1'})
        self.assertEqual(result,'complete')
        self.assertEqual(seen,{'cuda:0','cuda:1'})

    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.store = Store(self.temp.name)
        self.cfg = Config(rounds=2,hidden_size=8,warmup_transitions=1,batch_size=2,experimental_seller=3)
        make_experiment(self.store,'study',self.cfg,'mock',['ordinary','optimistic'],[1],1,1,False)

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def test_both_markets_issue_requests_concurrently_without_history_leakage(self):
        barrier = threading.Barrier(6)
        original = Gateway._mock
        def mock(gateway, position, seller, history):
            task, round_, _ = map(int,position.split('/'))
            if task in (0,1) and round_==1:
                barrier.wait(timeout=5)  # Fails if markets run sequentially.
            response = original(gateway,position,seller,history)
            if task in (0,1) and round_==1:
                response['output'][0]['content'][0]['text'] = json.dumps(
                    dict(price_index=15 if task==0 else 25,strategy=f'private-market-{task}'))
            return response
        with patch.object(Gateway,'_mock',mock):
            result = run_parallel(self.store,'study',POLICY,DEVICES)
        self.assertEqual(result,'complete')
        _, state, status = self.store.load('study')
        self.assertEqual(status,'complete')
        self.assertEqual(state['training_stop_reason'],'operational_convergence')
        self.assertEqual(len(state['tasks']),6)
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM requests').fetchone()[0],36)
        for task in (0,1):
            first = json.loads(self.store.records('study',task)[0]['record'])
            request = self.store.request('study',f'{task}/2/0')
            body = json.loads(request['body'])
            history = json.loads(body['input'][1]['content'])['complete_market_history']
            self.assertEqual(history[0][1:5],first['prices'])
            self.assertIn(f'private-market-{task}',request['body'])
            self.assertNotIn(f'private-market-{1-task}',request['body'])
        for key,learner in state['learners'].items():
            self.assertEqual(learner.steps,4)
            self.assertEqual(learner.updates,state['training_convergence']['monitors'][key]['history'][-1]['gradient_updates'])

    def test_migration_preserves_partial_game_weights_and_paid_request_identity(self):
        run_experiment(self.store,'study',progress=lambda _:None,step_limit=3)
        advance_training(self.store,'study',POLICY)
        values,state,status = self.store.load('study')
        history = state['market']['history'].copy()
        weights = {k:{n:v.clone() for n,v in l.online.state_dict().items()} for k,l in state['learners'].items()}
        requests = [tuple(r) for r in self.store.db.execute('SELECT id,position,body,response,charged FROM requests ORDER BY id')]
        initialize_parallel(state,DEVICES)
        self.assertEqual(state['parallel_execution']['markets'][1]['history'],history)
        self.assertEqual(state['parallel_execution']['completed_tasks'],[0])
        self.assertEqual(state['parallel_execution']['cycle_target'],2)
        for key,learner in state['learners'].items():
            for name,tensor in learner.online.state_dict().items():
                torch.testing.assert_close(tensor,weights[key][name])
        self.store.save('study',self.cfg,state,status)
        self.assertEqual(run_parallel(self.store,'study',POLICY,DEVICES),'complete')
        after = [tuple(self.store.db.execute('SELECT id,position,body,response,charged FROM requests WHERE id=?',(r[0],)).fetchone()) for r in requests]
        self.assertEqual(requests,after)
        self.assertEqual(len(self.store.records('study',1)),2)

    def test_worker_failure_drains_other_calls_and_does_not_silently_retry(self):
        barrier = threading.Barrier(6)
        original = Gateway._mock
        def mock(gateway,position,seller,history):
            barrier.wait(timeout=5)
            if position=='0/1/0':
                raise RuntimeError('Simulated transport failure')
            time.sleep(.02)
            return original(gateway,position,seller,history)
        with patch.object(Gateway,'_mock',mock):
            result = run_parallel(self.store,'study',POLICY,DEVICES)
        self.assertTrue(result.startswith('paused'))
        self.assertEqual(self.store.db.execute("SELECT count(*) FROM requests WHERE status='pending'").fetchone()[0],0)
        self.assertEqual(self.store.db.execute("SELECT count(*) FROM requests WHERE status='uncertain'").fetchone()[0],1)
        with patch.object(Gateway,'_mock') as send:
            with self.assertRaises(Paused):
                run_parallel(self.store,'study',POLICY,DEVICES)
            send.assert_not_called()

    def test_six_parallel_reservations_share_one_budget(self):
        self.store.budget(.10)
        def reserve(index):
            store = Store(self.temp.name)
            try:
                try:
                    store.reserve('study',f'{index//3}/1/{index%3}',{},.02)
                    return True
                except Paused:
                    return False
            finally:
                store.close()
        with ThreadPoolExecutor(max_workers=6) as pool:
            self.assertEqual(sum(pool.map(reserve,range(6))),5)
        self.assertAlmostEqual(self.store.spending()['reserved'],.10)
