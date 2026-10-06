import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from kazstyle.data.corpus import write_json
from kazstyle.training.suite import build_jobs,run


class FrozenSuiteTests(unittest.TestCase):
    def test_gpu_options_are_explicit_in_saved_commands(self):
        plan={'dataset':'data','seeds':[42],'models':[{'id':'encoder','model_name':'local/model','head':'last','batch_size':1,'gradient_accumulation':8}],
              'epochs':2,'learning_rate':2e-5,'threads':4,'device':'cuda','precision':'float16','gradient_checkpointing':True,'fused_adamw':True}
        command=build_jobs(plan)[1]['command']
        self.assertIn('--gradient-checkpointing',command);self.assertIn('--fused-adamw',command)
        self.assertEqual(command[command.index('--device')+1],'cuda')
        self.assertEqual(command[command.index('--precision')+1],'float16')

    def test_resume_does_not_retrain_completed_run_and_rejects_changed_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'kazstyle').mkdir();(root/'manage.py').write_text('# fixture')
            plan_path=root/'plan.json';out=root/'suite'
            plan={'dataset':'data','manifest_sha256':'manifest','run_prefix':'fixture','seeds':[42,43],'models':[]}
            write_json(plan_path,plan)
            launched=[]
            def execute(command,**kwargs):
                target=root/command[command.index('--out-dir')+1];target.mkdir(parents=True)
                write_json(target/'results.json',{'test_fixture':True});launched.append(target.name)
                return SimpleNamespace(returncode=0)
            with patch('kazstyle.training.suite.PROJECT_ROOT',root),patch('kazstyle.training.suite.artifact_path',side_effect=lambda name:root/'artifacts'/name),patch('kazstyle.training.suite.load_manifest',return_value=(None,{'manifest_sha256':'manifest'})),patch('kazstyle.training.suite.subprocess.run',side_effect=execute):
                self.assertEqual(run(plan_path,out,max_jobs=1)['status'],'partial')
                changed={**plan,'seeds':[42,43,44]};write_json(plan_path,changed)
                with self.assertRaisesRegex(ValueError,'changed plan'):run(plan_path,out,resume=True)
                write_json(plan_path,plan)
                self.assertEqual(run(plan_path,out,resume=True)['status'],'complete')
            self.assertEqual(launched,['fixture_classical_s42','fixture_classical_s43'])


if __name__=='__main__':unittest.main()
