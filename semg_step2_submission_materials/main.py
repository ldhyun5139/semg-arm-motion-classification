from pathlib import Path
import os, sys, json, time, platform, subprocess, shutil, zipfile
from itertools import combinations
import numpy as np
import pandas as pd
import scipy
from scipy.signal import butter, iirnotch, filtfilt
from scipy.stats import ttest_rel
import sklearn
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)
import xgboost
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
try:
    from IPython.display import display
except ImportError:
    def display(x): print(x)

ROOT = (Path('/content/semg_step2') if '__file__' not in globals()
        and Path('/content').exists() else Path.cwd())
ROOT.mkdir(parents=True, exist_ok=True)
RESULTS = ROOT / 'results'; FIGURES = ROOT / 'figures'
RESULTS.mkdir(exist_ok=True); FIGURES.mkdir(exist_ok=True)
FS, WIN, HOP = 1000, 200, 100
TRIAL_LEN, SEG_START, SEG_LEN = 5000, 1500, 2000
SEED = 42
DATA_URL = 'https://github.com/TestUser-alt-dev/sEMG-Hand-Gesture-Elbow-Angle-Dataset.git'
DATA_REVISION = 'cb64ded658b83ae97e6388dd373efad0e97bc47c'
CHANNELS = [f'bandpass {i}' for i in range(1, 6)]
CONFIG = dict(fs=FS, window=WIN, hop=HOP, trial_len=TRIAL_LEN,
              segment_offset_from_mark=SEG_START, segment_len=SEG_LEN,
              seed=SEED, data_revision=DATA_REVISION,
              segmentation='mark rising edge + 1.5s for 2s (assumption)',
              filtering='additional notch60 Q30 then bandpass20-450 order4',
              cv_scope='all trials, separately from hold-out; not external test',
              macro_average=True, cv_std_ddof=0)
(RESULTS/'config.json').write_text(json.dumps(CONFIG, indent=2), encoding='utf-8')
print('출력 폴더:', ROOT)

SOURCE = ROOT / '_source_data'
if not SOURCE.exists():
    subprocess.run(['git', 'clone', '--depth', '1', DATA_URL, str(SOURCE)], check=True)
actual_revision = subprocess.check_output(
    ['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip()
if actual_revision != DATA_REVISION:
    subprocess.run(['git', '-C', str(SOURCE), 'fetch', '--depth', '1',
                    'origin', DATA_REVISION], check=True)
    subprocess.run(['git', '-C', str(SOURCE), 'checkout', '--detach',
                    DATA_REVISION], check=True)
files = sorted((SOURCE/'data').rglob('*.csv'))
assert len(files) == 64, f'예상 64개 파일, 실제 {len(files)}개: 경로 확인'

def file_label(p):
    subject, rest = p.stem.split('_', 1)
    angle, gesture = rest.split()
    gesture = {'fist':'fist', 'open':'open',
               'pro':'pronation', 'extor':'supination'}[gesture]
    return subject, f'{angle}_{gesture}'

print('파일 수:', len(files)); print('데이터 revision:', DATA_REVISION)
print(pd.Series([file_label(p)[1] for p in files]).value_counts().sort_index())

def event_starts(df):
    m = pd.to_numeric(df['mark'], errors='raise').to_numpy() != 0
    return np.flatnonzero(m & ~np.r_[False, m[:-1]])

example = pd.read_csv(files[0])
starts = event_starts(example)
print('파일:', files[0].name, '배열:', example.shape)
print('mark 기준점 개수:', len(starts))
print('처음 5개 기준점(행 번호):', starts[:5])
print('기준점 간격(초):', np.unique(np.diff(starts) / FS))
print('onset 값별 개수:', example['onset'].value_counts().to_dict())
t = np.arange(min(len(example), 12000)) / FS
fig, axes = plt.subplots(5, 1, figsize=(12, 8), sharex=True)
for ch, ax in enumerate(axes):
    ax.plot(t, example[CHANNELS[ch]].to_numpy()[:len(t)], lw=0.5)
    for start in starts[starts < len(t)]:
        ax.axvline(start / FS, color='red', ls='--', lw=0.7)
        ax.axvspan((start+SEG_START)/FS, (start+SEG_START+SEG_LEN)/FS,
                   color='orange', alpha=0.15)
    ax.set_ylabel(CHANNELS[ch])
axes[-1].set_xlabel('Seconds from file start')
fig.suptitle('Red: event reference / Orange: assumed analysis segment')
fig.tight_layout(); fig.savefig(FIGURES/'event_alignment.png', dpi=160)
plt.show()

def preprocess(x, fs=FS):
    bn, an = iirnotch(60, 30, fs)
    x = filtfilt(bn, an, x, axis=0)
    b, a = butter(4, [20/(fs/2), 450/(fs/2)], btype='band')
    return filtfilt(b, a, x, axis=0)

def make_windows(x):
    return np.stack([x[i:i+WIN] for i in range(0, len(x)-WIN+1, HOP)])

def td_features(w, th=0.01):
    rms = np.sqrt(np.mean(w**2, axis=0))
    mav = np.mean(np.abs(w), axis=0)
    d = np.diff(w, axis=0)
    wl = np.sum(np.abs(d), axis=0)
    zc = np.sum((w[:-1]*w[1:] < 0) & (np.abs(d) >= th), axis=0)
    ssc = np.sum(d[:-1]*d[1:] < 0, axis=0)
    return np.concatenate([rms, mav, wl, zc, ssc])

X_parts, metadata, audit = [], [], []
for file_index, path in enumerate(files):
    df = pd.read_csv(path)
    subject, label = file_label(path)
    s = event_starts(df)
    dt = np.diff(df['time'].to_numpy(dtype=float))
    assert np.allclose(dt, 1/FS, atol=1e-7), f'{path.name}: time 간격이 일정하지 않습니다'
    assert len(s) == 30, f'{path.name}: mark {len(s)}개입니다. 예상 30개와 다르므로 확인이 필요합니다.'
    assert np.all(np.abs(np.diff(s)-TRIAL_LEN) <= 2), f'{path.name}: mark 간격을 확인하세요'
    signal = df[CHANNELS].to_numpy(dtype=np.float64)
    assert np.isfinite(signal).all(), f'{path.name}: NaN/Inf'
    audit.append(dict(file=path.name, subject=subject, label=label,
                      rows=len(df), events=len(s), first_event=int(s[0]),
                      last_event=int(s[-1])))
    for trial_no, start in enumerate(s):
        assert start+TRIAL_LEN <= len(signal), f'{path.name}: 마지막 5초 구간이 부족합니다'
        end = min(start+TRIAL_LEN, int(s[trial_no+1])) if trial_no+1 < len(s) else start+TRIAL_LEN
        trial = signal[start:end]
        seg = preprocess(trial)[SEG_START:SEG_START+SEG_LEN]
        windows = make_windows(seg)
        assert windows.shape == (19, 200, 5)
        features = np.stack([td_features(w) for w in windows])
        assert features.shape == (19, 25) and np.isfinite(features).all()
        trial_id = f'{path.stem}__trial{trial_no+1:02d}'
        X_parts.append(features)
        metadata.append(dict(trial_id=trial_id, subject=subject, label=label,
                             file=path.name, event_sample=int(start)))
    if (file_index+1) % 8 == 0:
        print(f'{file_index+1}/{len(files)} 파일 특징 생성 완료')

X = np.stack(X_parts)  # (trial, window, feature)
meta = pd.DataFrame(metadata)
le = LabelEncoder(); labels = le.fit_transform(meta['label'])
classes = le.classes_; class_ids = np.arange(len(classes))
assert X.shape == (1920, 19, 25) and len(classes) == 8
assert meta['trial_id'].is_unique
meta.to_csv(RESULTS/'trial_metadata.csv', index=False)
pd.DataFrame(audit).to_csv(RESULTS/'data_audit.csv', index=False)
np.savez_compressed(RESULTS/'features_unscaled.npz', X=X, y=labels,
                    trial_id=meta.trial_id.to_numpy(dtype=str), classes=classes)
print('특징 배열:', X.shape, '총 윈도우:', X.shape[0]*X.shape[1])
display(meta.groupby('label').size().rename('trial_count').to_frame())

tr_i, te_i = train_test_split(np.arange(len(meta)), test_size=0.2,
                             stratify=labels, random_state=SEED)
assert set(tr_i).isdisjoint(te_i)
def flatten(indices):
    return X[indices].reshape(-1, 25), np.repeat(labels[indices], X.shape[1])
Xtr, ytr = flatten(tr_i); Xte, yte = flatten(te_i)
te_meta = meta.iloc[te_i].reset_index(drop=True)
holdout_manifest = meta.copy()
holdout_manifest['split'] = 'train'
holdout_manifest.loc[te_i, 'split'] = 'test'
holdout_manifest.to_csv(RESULTS/'holdout_manifest.csv', index=False)
summary = pd.DataFrame({'split':['train','test'], 'trials':[len(tr_i),len(te_i)],
                         'windows':[len(ytr),len(yte)], 'features':[25,25]})
display(summary); summary.to_csv(RESULTS/'dataset_summary.csv', index=False)
counts = pd.DataFrame({'train':np.bincount(ytr, minlength=8),
                       'test':np.bincount(yte, minlength=8)}, index=classes)
display(counts); counts.to_csv(RESULTS/'class_distribution.csv')
shared_subjects = sorted(set(meta.iloc[tr_i].subject) & set(te_meta.subject))
print('학습·테스트에 공통으로 포함된 참가자:', shared_subjects)

models = {
 'SVM': Pipeline([('scaler', StandardScaler()), ('model',
     SVC(kernel='rbf', C=1.0, gamma='scale', decision_function_shape='ovr'))]),
 'RF': Pipeline([('scaler', StandardScaler()), ('model',
     RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1))]),
 'XGB': Pipeline([('scaler', StandardScaler()), ('model',
     XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05,
       subsample=0.9, colsample_bytree=0.8, random_state=SEED, n_jobs=2,
       eval_metric='mlogloss'))]),
}
def metrics(y, pred):
    return {'Accuracy':accuracy_score(y, pred),
      'Precision':precision_score(y, pred, labels=class_ids, average='macro', zero_division=0),
      'Recall':recall_score(y, pred, labels=class_ids, average='macro', zero_division=0),
      'F1-score':f1_score(y, pred, labels=class_ids, average='macro', zero_division=0)}

fitted, predictions, rows, costs = {}, {}, [], []
for name, prototype in models.items():
    model = clone(prototype)
    t0 = time.perf_counter(); model.fit(Xtr, ytr)
    train_seconds = time.perf_counter()-t0
    pred = model.predict(Xte)
    fitted[name], predictions[name] = model, pred
    rows.append({'Model':name, **metrics(yte, pred)})
    for _ in range(10): model.predict(Xte[:1])
    t0 = time.perf_counter()
    for _ in range(100): model.predict(Xte[:1])
    infer_ms = (time.perf_counter()-t0)/100*1000
    costs.append({'Model':name, 'Train_seconds':train_seconds,
                  'Inference_ms_per_window':infer_ms})
    print(name, '학습 및 평가 완료')
holdout = pd.DataFrame(rows).set_index('Model')
cost_table = pd.DataFrame(costs).set_index('Model')
display((holdout*100).round(2)); display(cost_table.round(4))
holdout.to_csv(RESULTS/'holdout_metrics.csv')
cost_table.to_csv(RESULTS/'computation_cost.csv')
prediction_table = pd.DataFrame({'trial_id':np.repeat(te_meta.trial_id.to_numpy(),19),
    'subject':np.repeat(te_meta.subject.to_numpy(),19), 'window':np.tile(np.arange(19),len(te_i)),
    'true':le.inverse_transform(yte)})
for name,pred in predictions.items(): prediction_table[name]=le.inverse_transform(pred)
prediction_table.to_csv(RESULTS/'holdout_predictions.csv',index=False)

error_rows, class_rows = [], []
fig, axes = plt.subplots(1,3,figsize=(24,7))
cms = {}
for ax,(name,pred) in zip(axes,predictions.items()):
    cm = confusion_matrix(yte,pred,labels=class_ids); cms[name]=cm
    np.savetxt(RESULTS/f'confusion_{name}.csv',cm,fmt='%d',delimiter=',')
    ax.imshow(cm,cmap='Blues'); ax.set_title(name)
    ax.set_xticks(class_ids,classes,rotation=65,ha='right'); ax.set_yticks(class_ids,classes)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    for i in class_ids:
        for j in class_ids:
            if cm[i,j]: ax.text(j,i,str(cm[i,j]),ha='center',va='center',
                fontsize=8,color='white' if cm[i,j]>cm.max()/2 else 'black')
        class_rows.append({'Model':name,'class':classes[i],
            'support':int(cm[i].sum()),'correct':int(cm[i,i]),
            'recall':float(cm[i,i]/cm[i].sum())})
        for j in class_ids:
            if i==j or cm[i,j]==0:continue
            ai,gi=classes[i].split('_',1); aj,gj=classes[j].split('_',1)
            kind=('angle_only' if gi==gj else 'gesture_only' if ai==aj else 'both')
            error_rows.append({'Model':name,'true':classes[i],'predicted':classes[j],
                               'count':int(cm[i,j]),'type':kind})
fig.tight_layout(); fig.savefig(FIGURES/'confusion8.png',dpi=180); plt.show()
for name,cm in cms.items():
    fig,ax=plt.subplots(figsize=(9,8)); ax.imshow(cm,cmap='Blues')
    ax.set_xticks(class_ids,classes,rotation=60,ha='right'); ax.set_yticks(class_ids,classes)
    ax.set(xlabel='Predicted',ylabel='True',title=name)
    for i in class_ids:
        for j in class_ids:
            ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=8,
                color='white' if cm[i,j]>cm.max()/2 else 'black')
    fig.tight_layout();fig.savefig(FIGURES/f'confusion_{name}.png',dpi=180);plt.close(fig)
errors=pd.DataFrame(error_rows,columns=['Model','true','predicted','count','type'])
per_class=pd.DataFrame(class_rows)
errors.to_csv(RESULTS/'error_pairs.csv',index=False)
per_class.to_csv(RESULTS/'class_recall.csv',index=False)
for name in models:
    print('\n',name,'클래스별 재현율');display(per_class[per_class.Model==name])
    print('오류가 많은 방향별 클래스 쌍')
    display(errors[errors.Model==name].sort_values('count',ascending=False).head(5))
error_types=(errors.groupby(['Model','type'])['count'].sum().unstack(fill_value=0)
             .reindex(index=list(models),columns=['angle_only','gesture_only','both'],fill_value=0))
display(error_types);error_types.to_csv(RESULTS/'error_types.csv')

skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=SEED)
folds=list(skf.split(np.zeros(len(meta)),labels))
fold_rows=[]; fold_acc={name:[] for name in models}
oof={name:np.full((len(meta),19),-1,dtype=int) for name in models}
fold_id=np.zeros(len(meta),dtype=int)
for fold,(a,b) in enumerate(folds,1):
    assert set(a).isdisjoint(b)
    assert set(meta.iloc[a].trial_id).isdisjoint(meta.iloc[b].trial_id)
    Xa,ya=flatten(a); Xb,yb=flatten(b); fold_id[b]=fold
    for name,prototype in models.items():
        m=clone(prototype);m.fit(Xa,ya);pred=m.predict(Xb)
        scores=metrics(yb,pred)
        fold_rows.append({'fold':fold,'Model':name,**scores})
        fold_acc[name].append(scores['Accuracy'])
        oof[name][b]=pred.reshape(-1,19)
        print(f'fold {fold}/5 {name}: acc={scores["Accuracy"]*100:.2f}%')
    pd.DataFrame(fold_rows).to_csv(RESULTS/'cv_folds.csv',index=False)
cv=pd.DataFrame(fold_rows)
cv_summary=cv.groupby('Model')[list(holdout.columns)].agg(['mean',lambda s:s.std(ddof=0)])
cv_summary.columns=['_'.join([a,'mean' if b=='mean' else 'std']) for a,b in cv_summary.columns]
cv_summary=cv_summary.reindex(list(models));cv_summary.to_csv(RESULTS/'cv_summary.csv')
for name in models:
    assert (oof[name]>=0).all()
    print(f'{name}: {np.mean(fold_acc[name])*100:.2f} ± {np.std(fold_acc[name])*100:.2f}%')
display(cv.pivot(index='fold',columns='Model',values='Accuracy')*100)
manifest=meta.copy();manifest['validation_fold']=fold_id
manifest.to_csv(RESULTS/'cv_manifest.csv',index=False)
np.savez_compressed(RESULTS/'cv_oof_predictions.npz',y=labels,**oof)

test_rows=[]
for a,b in combinations(models,2):
    av,bv=np.array(fold_acc[a]),np.array(fold_acc[b]);diff=av-bv
    if not np.isfinite(diff).all() or np.isclose(diff.std(ddof=1),0,atol=1e-12):
        t,p=np.nan,np.nan
        interpretation='차이의 분산이 0이거나 값이 유한하지 않아 t 검정 해석이 어렵습니다.'
    else:
        t,p=ttest_rel(av,bv)
        interpretation=('이번 5개 폴드의 정확도 차이가 유의수준 0.05에서 유의합니다.'
                        if p<0.05 else '이번 5개 폴드에서 유의한 차이를 검출하지 못했습니다. 동등함을 뜻하지 않습니다.')
    test_rows.append({'A':a,'B':b,'mean_diff_percentage_points':diff.mean()*100,
                      't':t,'p_raw':p,'interpretation':interpretation})
tests=pd.DataFrame(test_rows);tests['p_holm']=np.nan
valid=tests.index[np.isfinite(tests.p_raw)]
if len(valid)==3:
    ordered=tests.loc[valid].sort_values('p_raw').index
    adjusted=np.maximum.accumulate([min(1,tests.loc[i,'p_raw']*(3-j))
                                    for j,i in enumerate(ordered)])
    tests.loc[ordered,'p_holm']=adjusted
tests.to_csv(RESULTS/'paired_ttests.csv',index=False)
display(tests)

trial_rows=[]; subject_rows=[]
for name,pred in predictions.items():
    trial_predictions=[]; ties=0
    for window_pred in pred.reshape(-1,19):
        counts=np.bincount(window_pred,minlength=8)
        ties+=int((counts==counts.max()).sum()>1)
        trial_predictions.append(int(np.argmax(counts)))
    trial_rows.append({'Model':name,'window_accuracy':accuracy_score(yte,pred),
        'trial_vote_accuracy':accuracy_score(labels[te_i],trial_predictions),'tie_trials':ties})
    subjects=np.repeat(te_meta.subject.to_numpy(),19)
    for subject in sorted(set(subjects)):
        mask=subjects==subject
        subject_rows.append({'Model':name,'subject':subject,'windows':int(mask.sum()),
                             **metrics(yte[mask],pred[mask])})
trial_comparison=pd.DataFrame(trial_rows);subject_comparison=pd.DataFrame(subject_rows)
display(trial_comparison);display(subject_comparison)
trial_comparison.to_csv(RESULTS/'trial_vote_comparison.csv',index=False)
subject_comparison.to_csv(RESULTS/'subject_metrics.csv',index=False)
environment={'python':sys.version,'platform':platform.platform(),
 'cpu':platform.processor(),'logical_cpu_count':os.cpu_count(),
 'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,
 'scikit-learn':sklearn.__version__,'xgboost':xgboost.__version__,
 'data_revision':DATA_REVISION,'seed':SEED,'threads_RF':-1,'threads_XGB':2}
(RESULTS/'environment.json').write_text(json.dumps(environment,indent=2),encoding='utf-8')
(RESULTS/'model_parameters.json').write_text(json.dumps(
 {n:m.named_steps['model'].get_params() for n,m in models.items()},indent=2,default=str),encoding='utf-8')


def md_table(frame):
    f=frame.reset_index() if frame.index.name else frame.copy()
    lines=['| '+' | '.join(map(str,f.columns))+' |', '| '+' | '.join(['---']*len(f.columns))+' |']
    for row in f.itertuples(index=False,name=None):
        lines.append('| '+' | '.join(str(v) for v in row)+' |')
    return '\n'.join(lines)

cv_text=pd.DataFrame({'Model':list(models), 'CV Accuracy mean ± std (%)':[
    f'{np.mean(fold_acc[n])*100:.2f} ± {np.std(fold_acc[n])*100:.2f}' for n in models]})
best_cv=max(fold_acc,key=lambda k:np.mean(fold_acc[k]))
holdout_text=(holdout*100).round(2).reset_index()
readme=f"""# sEMG 기반 손동작 및 팔꿈치 각도 분류
이동현 21113718 · 딥러닝프로그래밍 Step 2 최종 과제

> 실행 결과를 바탕으로 생성한 초안입니다. [직접 작성] 부분을 완성한 후 제출합니다.

## 1. 코드 설명
5채널 sEMG에서 4개 손동작과 2개 팔꿈치 각도를 조합한 8개 클래스를 분류합니다.
원본 CSV 64개에서 mark 기준으로 30개씩 구분한 시행 후보를 사용하였습니다.
mark를 5초 시행 후보의 기준점으로 사용하되, 실제 동작 시점의 의미는 확정하지 않았습니다.
각 기준점에서 1.5초 뒤부터 2초를 분석하는 것은 실습 가정입니다.
제공된 bandpass 신호에 강의의 노치60Hz Q30 및 20~450Hz 4차 필터를 추가 적용하였습니다.
200ms, 100ms 간격으로 시행당 19개 윈도우에서 RMS/MAV/WL/ZC/SSC를 계산하였습니다.
각 윈도우의 25개 특징을 별도 입력 표본으로 사용하였습니다.

전체 1,920개 시행 후보를 80:20으로 먼저 나누었습니다.
학습 1,536개 시행/29,184개 윈도우, 테스트 384개 시행/7,296개 윈도우입니다.
동일 시행 윈도우 중복은 차단하였으며, 같은 참가자 및 파일의 다른 시행은 공유될 수 있습니다.
표준화는 각 학습 집합에서만 fit하였습니다. 5-fold는 전체 시행에서 별도로 수행했습니다.
따라서 CV와 hold-out은 독립 외부 평가 두 개가 아닙니다. 지표는 Accuracy와 macro Precision/Recall/F1입니다.

### 실행 방법
```bash
pip install -r requirements.txt
python main.py
```
데이터는 고정 revision에서 자동 다운로드됩니다. CPU로 실행할 수 있습니다.
원 데이터: https://github.com/TestUser-alt-dev/sEMG-Hand-Gesture-Elbow-Angle-Dataset
데이터 revision: {DATA_REVISION}
데이터셋 인용 논문 DOI: 10.1007/s13534-026-00607-7

| 파일 | 역할 |
| --- | --- |
| main.py | 데이터 확보, 특징 생성, 학습, hold-out, CV, 검정, 결과 저장 |
| requirements.txt | 실행 시 사용한 주요 패키지 버전 |
| results/config.json | 전처리, 분할 및 실습 가정 |
| results/*manifest.csv | hold-out 및 CV의 시행별 배정 |
| results/holdout_predictions.csv | 재확인 가능한 실제/예측 라벨 |
| figures/ | 혼동행렬과 이벤트 정렬 그림 |
| report.md 및 report.pdf | 7개 항목의 최종 리포트(직접 완성 후 PDF 추가) |

## 2. 모델 성능 비교
다음 hold-out 지표의 단위는 %입니다.
{md_table(holdout_text)}

### 5-fold 교차검증
표준편차는 ddof=0이며 폴드별 값은 results/cv_folds.csv에 저장했습니다.
{md_table(cv_text)}

이번 5-fold에서 평균 정확도가 가장 높은 모델은 **{best_cv}**입니다.
[직접 작성] hold-out과 CV에서 모델 순위가 같은지, 평균 차이와 폴드별 변동이 어떠한지 설명합니다.
논문 보고값 RF 96.72±0.58%, XGB 96.15±0.56%, SVM 95.83±0.79%와는 분할/표본 구성 차이를 명시하고 비교합니다.

### 통계 검정
{md_table(tests.round(6))}
유의수준 0.05의 양측 대응표본 t 검정입니다. p_raw와 참고용 p_holm을 구분합니다.
5개 폴드의 표본 수와 중복 학습셋 때문에 해석에 제한이 있습니다.
p≥0.05를 세 모델의 동등성으로 해석하지 않습니다.
[직접 작성] 세 쌍의 평균 차이, p값, 검출 여부를 각각 문장으로 해석합니다.

### 계산 비용
{md_table(cost_table.round(4).reset_index())}
학습 시간은 scaler+fit입니다. 추론은 이미 계산한 25D 특징 한 개의 scaler+predict이며,
예열 10회 후 100회 평균입니다. 신호 취득, 필터, 특징 추출 및 데이터 다운로드는 제외합니다.
환경은 results/environment.json을 참고합니다. Colab의 시간은 장치와 실행 상황에 따라 달라집니다.

## 3. Confusion Matrix 분석
![세 모델 혼동행렬](figures/confusion8.png)
행=실제, 열=예측입니다. error_pairs.csv, class_recall.csv, error_types.csv에 수치를 저장했습니다.
"""
for name in models:
    pc=per_class[per_class.Model==name]
    best=pc.loc[pc.recall==pc.recall.max(),'class'].tolist()
    worst=pc.loc[pc.recall==pc.recall.min(),'class'].tolist()
    top=errors[errors.Model==name].sort_values('count',ascending=False).head(5)
    readme+=f"""\n### {name}
![{name} 혼동행렬](figures/confusion_{name}.png)
- 가장 잘 분류된 클래스(재현율 기준, 공동 포함): {', '.join(best)}
- 재현율이 가장 낮은 클래스(공동 포함): {', '.join(worst)}
- 주요 방향별 오류 쌍:
{md_table(top)}

[직접 작성] 각도만 틀린 오류와 동작만 틀린 오류 중 어느 쪽이 많은지 수치로 설명합니다.
[직접 작성] 가능한 원인과 추가 확인 방법을 적습니다. 혼동행렬만으로 근육이나 생리적 원인을 확정하지 않습니다.
"""
readme+=f"""\n### 추가 분석
{md_table(trial_comparison.round(4))}
윈도우 단위와 시행 다수결 정확도를 비교합니다. 다수결 동률은 인코딩 순서로 처리했습니다.
[직접 작성] 집계 단위가 바뀌면 왜 성능이 달라지는지, 높은 다수결 성능의 실제 의미가 무엇인지 설명합니다.

## 4. 최종 결과
- 이번 CV 평균 정확도가 가장 높은 모델: {best_cv}
- [직접 작성] 가장 낮은 모델, 주요 오류 클래스, 정확도와 비용을 함께 고려한 선택
- [직접 작성] 실험 결과로 알게 된 점과 아직 확인하지 못한 점

### 한계
1. Trial-wise 분할이므로 같은 사람/같은 기록 파일의 다른 시행이 학습과 평가에 함께 들어갈 수 있습니다. 새 피험자 일반화는 LOSO 등 별도 검증이 필요합니다.
2. mark 기준의 분석 시작점 +1.5초는 가정이며 실제 근활성 구간 정렬을 확정하지 않았습니다.
3. 제공 데이터는 이미 필터링되어 있으며 이번 실습은 추가 필터를 적용했습니다. 논문의 처리와 정확히 동일하다고 주장하지 않습니다.
4. 19개 겹친 윈도우는 독립된 19회 실험이 아닙니다. 윈도우 정확도와 시행 정확도를 구분합니다.
5. 5-fold p값은 작은 표본과 폴드 간 의존성의 영향을 받습니다. 유의하지 않음은 동등성의 증거가 아닙니다.
6. Colab 계산 비용은 특정 실행 환경의 결과이며, 임베디드 실시간 시스템의 전체 지연을 뜻하지 않습니다.

## 재현 정보
분할/모델 seed=42, CV=5-fold stratified, 주요 설정은 results/config.json 및 model_parameters.json,
환경 버전은 results/environment.json, 분할 목록은 manifest 파일에 기록했습니다.
"""
(ROOT/'README.md').write_text(readme,encoding='utf-8')
report="""# Step 2 최종 리포트
이동현 21113718

## 1 문제 정의
[직접 작성] 5채널로 손동작 4종과 각도 2종을 동시에 분류하는 목적, 8클래스 정의
## 2 데이터
[직접 작성] 출처·revision·64파일/1,920시행 후보·분할·mark 가정·5주차와 달라진 범위
## 3 방법
[직접 작성] 필터·윈도우·특징·표준화·세 모델 하이퍼파라미터
## 4 결과
[직접 작성] hold-out 4지표 표·5-fold 폴드별 값과 평균±표준편차·세 쌍 t검정 해석·계산 비용
## 5 오류 분석
[직접 작성] 모델별 혼동행렬·최대 오류 쌍·각도/동작 오류 구조·시행 다수결·원인 가설과 검증 방법
## 6 한계
[직접 작성] 최소 3개, trial-wise/피험자 공유·정렬 가정·필터 중복·윈도우 의존성 포함
## 7 재현 정보
[직접 작성] 환경 버전·시드·실행 순서·데이터 revision·README와 결과 파일 경로
"""
(ROOT/'report.md').write_text(report,encoding='utf-8')
versions={'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,
          'scikit-learn':sklearn.__version__,'xgboost':xgboost.__version__}
import matplotlib
versions['matplotlib']=matplotlib.__version__
(ROOT/'requirements.txt').write_text(''.join(f'{k}=={v}\n' for k,v in versions.items()),encoding='utf-8')
(ROOT/'.gitignore').write_text('_source_data/\n__pycache__/\n.venv/\n.ipynb_checkpoints/\n',encoding='utf-8')
print('생성 완료:',ROOT)
print('주의: README/report의 [직접 작성] 항목은 아직 미완성입니다.')
