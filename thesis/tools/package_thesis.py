"""Build an auditable, self-contained Overleaf ZIP from actual LaTeX inputs.

Run with a Python environment containing pypdf and a recent pdflatex on PATH.
This validates compilation and the clean ZIP build, not scientific or visual QA.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import zipfile
from pypdf import PdfReader


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def compile_document(source, build, compiler):
    build.mkdir(parents=True,exist_ok=False)
    for i in range(3):
        with (build/f'pass-{i+1}.txt').open('w') as log:
            subprocess.run([compiler,'-no-shell-escape','-interaction=nonstopmode',
                '-halt-on-error','-file-line-error','-recorder',
                '-output-directory='+str(build),'main.tex'],cwd=source,
                stdout=log,stderr=subprocess.STDOUT,check=True,timeout=120)
    log=(build/'main.log').read_text(errors='replace')
    bad=[line for line in log.splitlines() if re.search(
        r'Overfull \\[hv]box|undefined|multiply defined|Rerun to get|Label\(s\) may have changed',line)]
    if bad:raise RuntimeError('Resolve LaTeX warnings before delivery: '+repr(bad))
    return build/'main.pdf'


def page_text(pdf):
    return [p.extract_text() for p in PdfReader(pdf).pages]


def main(args):
    source=args.source.resolve();output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    compiler=shutil.which(args.compiler)
    if compiler is None:raise FileNotFoundError(args.compiler)
    with tempfile.TemporaryDirectory(prefix='cnr-thesis-') as temp:
        temp=Path(temp);build=temp/'build'
        try:
            pdf=compile_document(source,build,compiler)
        except Exception:
            if build.exists():shutil.copytree(build,output/'failed-build')
            raise
        inputs=set()
        for line in (build/'main.fls').read_text().splitlines():
            if not line.startswith('INPUT '):continue
            p=Path(line[6:]);p=(source/p).resolve() if not p.is_absolute() else p.resolve()
            if p.is_file() and p.is_relative_to(source):
                rel=p.relative_to(source)
                if p.suffix not in ('.aux','.toc','.lof','.lot','.out','.log','.fls'):
                    inputs.add(rel)
        assert Path('main.tex') in inputs and Path('metadata.tex') in inputs
        for name in ('README.md','TEMPLATE.md'):
            if (source/name).is_file():inputs.add(Path(name))
        archive=output/f'PoliTo_Thesis_{args.version}_Overleaf.zip'
        with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
            for rel in sorted(inputs):z.write(source/rel,str(rel))
        clean=temp/'clean';clean.mkdir()
        with zipfile.ZipFile(archive) as z:z.extractall(clean)
        try:
            rebuilt=compile_document(clean,temp/'clean-build',compiler)
        except Exception:
            if (temp/'clean-build').exists():
                shutil.copytree(temp/'clean-build',output/'failed-clean-build')
            raise
        texts=page_text(pdf)
        assert texts==page_text(rebuilt),'Clean package build changes PDF page text'
        final=output/f'Road_Conditioned_Diffusion_Thesis_{args.version}.pdf'
        shutil.copy2(pdf,final)
        aux=(build/'main.aux').read_text();toc=(build/'main.toc').read_text()
        cited=set()
        for entry in re.findall(r'\\citation\{([^}]+)\}',aux):cited.update(entry.split(','))
        bibliography=set(re.findall(r'\\bibcite\{([^}]+)\}',aux))
        assert cited<=bibliography
        appendix_pages=re.findall(r'\}\{(\d+)\}\{appendix\.',toc)
        receipt={'version':args.version,'pdf_pages':len(texts),
            'main_text_pages':int(appendix_pages[0])-1 if appendix_pages else None,
            'figures':(build/'main.lof').read_text().count('\\contentsline {figure}'),
            'tables':(build/'main.lot').read_text().count('\\contentsline {table}'),
            'cited_bibliography_entries':len(cited),'bibliography_entries':len(bibliography),
            'clean_zip_build_page_text_identical':True,'latex_overfull_or_undefined':0,
            'visual_review_complete':False,'scientific_review_complete':False,
            'compiler':subprocess.check_output([compiler,'--version'],text=True).splitlines()[0],
            'inputs':{str(p):{'sha256':digest(source/p),'bytes':(source/p).stat().st_size} for p in sorted(inputs)},
            'outputs':{p.name:{'sha256':digest(p),'bytes':p.stat().st_size} for p in (final,archive)},
            'builder_sha256':digest(__file__)}
        for suffix in ('log','toc','lof','lot'):
            shutil.copy2(build/('main.'+suffix),output/('build.'+suffix))
        (output/'build-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
        print(json.dumps({k:v for k,v in receipt.items() if k not in ('inputs','outputs')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--version',required=True)
    p.add_argument('--compiler',default='pdflatex');main(p.parse_args())
