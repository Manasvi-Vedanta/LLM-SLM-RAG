import json
from pathlib import Path
from evaluate import TEST_QUESTIONS, run_evaluation, compute_metrics
from critic import GemmaCritic

models = ['gemma3-critic', 'gemma3-critic-v2', 'gemma3-critic-v3']
summary = {}
all_results = []

for model in models:
    critic = GemmaCritic(model_name=model)
    results = run_evaluation({model: critic}, TEST_QUESTIONS)
    all_results.extend(results)
    summary[model] = compute_metrics(results, model)

Path('benchmark_v123_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
Path('benchmark_v123_results.json').write_text(json.dumps([r.__dict__ for r in all_results], indent=2), encoding='utf-8')
print(json.dumps(summary, indent=2))
