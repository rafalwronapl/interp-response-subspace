import sys
from analysis.Qwen06_ABC_extension_20261102.b_worker import run,ROOT,receipt,read,sha
from analysis.Qwen06_ABC_extension_20261102.b_score import analyze
if __name__=='__main__':
 import argparse
 parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);cfg=parser.parse_args().config;stage=(ROOT/cfg).parent
 if (stage/'summary.json').exists():assert receipt(stage/'summary.json')['source_sha256']==sha(ROOT/read(stage/'config.json')['output']/'result.json')
 else:run(cfg);analyze(stage)
