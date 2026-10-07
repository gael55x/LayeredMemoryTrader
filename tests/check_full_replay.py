"""Check the real loader, encoder, FAISS and replay with constructed Gemini responses.

Run from the repository: python tests/check_full_replay.py
Requires requirements.txt. First run downloads the public embedding model.
No Gemini credentials or billed calls; no trading-performance measurement.
"""
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

os.environ['GEMINI_API_KEY'] = 'offline-adoption-test-key'
os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = '1'
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "layered-trader-mpl"))
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
repo = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo))
import httpx
import pandas as pd
import yaml
import matplotlib
matplotlib.use('Agg')
from google import genai
from trader import Trader
from evaluate import evaluate_performance

config = yaml.safe_load((repo/'config.yaml').read_text())
source = pd.read_csv(repo/'historical_data_with_news.csv', parse_dates=['time'])
period = config['backtest']['training_period']
sample = source[source['time'].between(period['start'], period['end'])].groupby('ticker', sort=False).head(25)
assert len(sample) == 75
requests = []
real_client = genai.Client

def handler(request):
    requests.append(request.url.path)
    return httpx.Response(200, json={'candidates':[{'content':{'role':'model','parts':[{'text':'VOTE: HOLD, CONFIDENCE: 0.9'}]},'finishReason':'STOP'}]})

@contextlib.contextmanager
def offline_client(**kwargs):
    with httpx.Client(transport=httpx.MockTransport(handler)) as transport:
        kwargs['http_options'] = kwargs['http_options'].model_copy(update={'httpx_client':transport})
        with real_client(**kwargs) as client:
            yield client

cwd = Path.cwd()
with tempfile.TemporaryDirectory(prefix='gael-trader-adoption-') as temp:
    path = Path(temp)
    sample.to_csv(path/'prices.csv', index=False)
    config['backtest']['news_data_path'] = str(path/'prices.csv')
    config['backtest']['full_data_path'] = str(path/'prices.csv')
    (path/'config.yaml').write_text(yaml.safe_dump(config))
    try:
        os.chdir(path)
        with patch.object(genai, 'Client', offline_client), contextlib.redirect_stdout(io.StringIO()):
            trader = Trader(config_path=str(path/'config.yaml'), backtest_mode='train')
            trader.run_backtest()
            assert len(trader.reflection_log) == 12
            assert trader.reflection_log['decision'].eq('HOLD').all()
            assert len(requests) == 27
            assert trader.semantic_memory.index.ntotal == len(trader.semantic_memory.entries)
            assert trader.semantic_memory.index.ntotal > 0
            assert trader.semantic_memory.model.encode(['adoption check']).shape == (1,384)
            evaluate_performance(trader)
        out = path/'documentation/results'
        assert len(pd.read_csv(out/'trade_log.csv')) == 12
        assert all((out/f).stat().st_size > 0 for f in ['decision_distribution_pie_chart.png','portfolio_performance.png','summary_report.md'])
        print(json.dumps({
            'sample_price_rows': 75,
            'tickers': 3,
            'decision_dates': 12,
            'constructed_sdk_requests': 27,
            'exported_csv_rows': 12,
            'real_embedding_dimensions': 384,
            'real_faiss_entries': trader.semantic_memory.index.ntotal,
            'evaluator_csv_charts_report_written': True,
            'scope': 'Actual CSV loader, constructor, encoder, FAISS, agents, debate, replay and evaluator. SDK HTTP response is constructed HOLD; no Gemini network, no measured trading performance.'
        }, indent=2))
    finally:
        os.chdir(cwd)
