import hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
from torchvision.models import resnet18,ResNet18_Weights
from scipy.optimize import minimize_scalar
from scipy.special import softmax,logsumexp
root=Path('g12-data')
def metrics(logits,y,classes=range(6)):
    p=softmax(logits,axis=1);pred=p.argmax(1);cm=np.zeros((6,6),int);np.add.at(cm,(y,pred),1)
    f=2*cm.diagonal()/np.maximum(1,cm.sum(0)+cm.sum(1));conf=p.max(1)
    bins=np.minimum(9,(10*conf).astype(int));ece=sum(np.mean(bins==i)*abs(np.mean(conf[bins==i])-np.mean(pred[bins==i]==y[bins==i])) for i in set(bins))
    return dict(acc=float(np.mean(pred==y)),f1=float(f[list(classes)].mean()),byclass=f.tolist(),cm=cm.tolist(),nll=float(np.mean(logsumexp(logits,axis=1)-logits[np.arange(len(y)),y])),ece=float(ece))
def model():
    m=resnet18(weights=ResNet18_Weights.IMAGENET1K_V1);m.fc=torch.nn.Linear(512,6);return m.cuda()
def features(img):
    mean=torch.tensor([.485,.456,.406],device='cuda');std=torch.tensor([.229,.224,.225],device='cuda')
    return ((img.float()/255-mean)/std).permute(0,3,1,2)
@torch.no_grad()
def predict(m,x):
    m.eval();return torch.cat([m(features(z)) for z in x.split(48)]).cpu().numpy()
def train(seed,recipe="plain"):
    torch.set_num_threads(2);torch.manual_seed(seed);rng=np.random.default_rng(203110)
    path=Path('data/processed/gaf-duration-e2/length512');z=np.load(path/'train.npz');val=np.load(path/'val.npz')
    y=z['labels'];ids=z['ids'];assert len(set(ids))==len(ids) and set(ids).isdisjoint(val['ids'])
    sets=[[],[],[]]
    for c in range(6):
        ix=np.flatnonzero(y==c);ix=ix[np.argsort(ids[ix])];rng.shuffle(ix);n=max(1,round(len(ix)*.1))
        for group,part in zip(sets,[ix[2*n:],ix[:n],ix[n:2*n]]):group.extend(part)
    fit,sel,cal=map(np.array,sets)
    x=torch.from_numpy(z['images']).cuda();ty=torch.tensor(y,device='cuda');vx=torch.from_numpy(val['images']).cuda()
    m=model()
    if recipe=='regular':
        opt=torch.optim.AdamW([{'params':[v for k,v in m.named_parameters() if not k.startswith('fc.')],'lr':5e-5},{'params':m.fc.parameters(),'lr':3e-4}],weight_decay=1e-4)
        sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=50)
    else:
        opt=torch.optim.AdamW(m.parameters(),lr=2e-4,weight_decay=1e-4)
    prefix='' if recipe=='plain' else 'reg-'
    w=torch.bincount(ty[fit],minlength=6).float().rsqrt();w/=w.mean()
    best=-1;log=[]
    for epoch in range(1,51):
        m.train();order=fit[torch.randperm(len(fit)).numpy()]
        for ix in np.array_split(order,int(np.ceil(len(order)/64))):
            opt.zero_grad();loss=torch.nn.functional.cross_entropy(m(features(x[ix])),ty[ix],weight=w,label_smoothing=.05 if recipe=="regular" else 0.);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1);opt.step()
        if recipe=='regular':sched.step()
        v=metrics(predict(m,x[sel]),y[sel]);log.append([epoch,v['f1'],v['nll']])
        if v['f1']>best:
            best=v['f1'];chosen=epoch;torch.save(m.state_dict(),root/f'{prefix}scorer{seed}.pt')
    final=metrics(predict(m,vx),val['labels'])
    m.load_state_dict(torch.load(root/f'{prefix}scorer{seed}.pt',weights_only=True));lc=predict(m,x[cal]);lv=predict(m,vx)
    temp=float(np.exp(minimize_scalar(lambda a:metrics(lc/np.exp(a),y[cal])['nll'],bounds=(-3,3),method='bounded').x))
    selmetrics=metrics(lv,val['labels']);calmetrics=metrics(lv/temp,val['labels'])
    m.load_state_dict(torch.load(f'runs/gaf-duration-e2/length512-seed{seed}/final.pt',weights_only=True));old=metrics(predict(m,vx),val['labels'])
    (root/f'{prefix}fit{seed}.json').write_text(json.dumps(dict(seed=seed,n=[len(g) for g in sets],split_hash=[hashlib.sha256('\n'.join(sorted(map(str,ids[g]))).encode()).hexdigest() for g in sets],epoch=chosen,T=temp,curve=log,old=old,final=final,selected=selmetrics,calibrated=calmetrics),indent=1))
if __name__=='__main__':train(int(sys.argv[1]),sys.argv[2] if len(sys.argv)>2 else "plain")
