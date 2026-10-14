"""Unattenuated learned outputs, frozen games and development-only search/matches."""
from games.connect4.alphazero_v2.experimental import rng_payload, restore_rng, digest
from research.alphazero_resnet_selfplay.measures import score_examples, summary, value_reliability
from research.alphazero_resnet_selfplay.evaluate import fixtures, openings, raw_record
from research.alphazero_value_diagnostics.evaluate import search, matches as prior_matches
from .common import RESULTS, read, write
from .checkpoint import state
from .data import datasets, enforce_membership


def values(model, rows, examples):
    scored,_,_,predictions=score_examples(model,examples)
    records=[dict(r,value=float(v),se=float((v-r['target'])**2)) for r,v in zip(rows,predictions)]
    out=summary(scored)
    out.update(mean_value=float(predictions.mean()),mean_target=sum(r['target'] for r in rows)/len(rows),
               bias=sum(r['value']-r['target'] for r in records)/len(records))
    return dict(summary=out,rows=records)


def evaluate(runner, output, check):
    output.mkdir(parents=True,exist_ok=False)
    before=state(runner)
    global_before=rng_payload()
    try:
        seed=runner.config.training.seed
        (train,examples),(held,held_examples)=datasets(seed)
        enforce_membership(seed,train,held)
        check()
        value_records=dict(training=values(runner.model,train,examples),heldout=values(runner.model,held,held_examples))
        novel=[i for i,r in enumerate(held) if not r['mirror_overlap']]
        value_records['novel']=values(runner.model,[held[i] for i in novel],[held_examples[i] for i in novel])
        write(output/'values.json',value_records)
        inference=runner.factory.inference(runner.model)
        suites={}
        for suite in ('tactical','solved','deep'):
            records=[]
            for row in fixtures()[suite]:
                check()
                record=raw_record(inference,suite,row)
                record['search']=search(inference,suite,row,4404901)
                records.append(record)
            suites[suite]=records
        write(output/'positions.json',suites)
        check()
        # prior wrapper checks within matches and each turn; exact original opponent/search.
        matched=prior_matches(inference,dict(match_seed=4404902),None,check,rows=openings())
        write(output/'matches.json',dict(records=matched))
    finally:
        restore_rng(global_before)
    if digest(state(runner))!=digest(before):
        raise ValueError('Evaluation changed learner computational state')
    write(output/'complete.json',dict(complete=True,state_sha256=digest(before),evaluation_rng_neutral=True))
