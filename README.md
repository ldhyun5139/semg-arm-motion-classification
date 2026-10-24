# sEMG 기반 손동작 및 팔꿈치 각도 분류
이동현 21113718 · 딥러닝프로그래밍 Step 2 최종 과제


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
데이터 revision: cb64ded658b83ae97e6388dd373efad0e97bc47c
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
| Model | Accuracy | Precision | Recall | F1-score |
| --- | --- | --- | --- | --- |
| SVM | 47.82 | 48.57 | 47.82 | 47.89 |
| RF | 65.27 | 65.44 | 65.27 | 65.28 |
| XGB | 63.58 | 63.77 | 63.58 | 63.6 |

### 5-fold 교차검증
표준편차는 ddof=0이며 폴드별 값은 results/cv_folds.csv에 저장했습니다.
| Model | CV Accuracy mean ± std (%) |
| --- | --- |
| SVM | 47.57 ± 0.35 |
| RF | 64.54 ± 0.72 |
| XGB | 62.86 ± 0.83 |

이번 5-fold에서 평균 정확도가 가장 높은 모델은 **RF**입니다.
[직접 작성] hold-out과 CV에서 모델 순위가 같은지, 평균 차이와 폴드별 변동이 어떠한지 설명합니다.
논문 보고값 RF 96.72±0.58%, XGB 96.15±0.56%, SVM 95.83±0.79%와는 분할/표본 구성 차이를 명시하고 비교합니다.

### 통계 검정
| A | B | mean_diff_percentage_points | t | p_raw | interpretation | p_holm |
| --- | --- | --- | --- | --- | --- | --- |
| SVM | RF | -16.968202 | -50.646763 | 1e-06 | 이번 5개 폴드의 정확도 차이가 유의수준 0.05에서 유의합니다. | 3e-06 |
| SVM | XGB | -15.287829 | -46.848181 | 1e-06 | 이번 5개 폴드의 정확도 차이가 유의수준 0.05에서 유의합니다. | 3e-06 |
| RF | XGB | 1.680373 | 10.825422 | 0.000413 | 이번 5개 폴드의 정확도 차이가 유의수준 0.05에서 유의합니다. | 0.000413 |
유의수준 0.05의 양측 대응표본 t 검정입니다. p_raw와 참고용 p_holm을 구분합니다.
5개 폴드의 표본 수와 중복 학습셋 때문에 해석에 제한이 있습니다.
p≥0.05를 세 모델의 동등성으로 해석하지 않습니다.
[직접 작성] 세 쌍의 평균 차이, p값, 검출 여부를 각각 문장으로 해석합니다.

### 계산 비용
| Model | Train_seconds | Inference_ms_per_window |
| --- | --- | --- |
| SVM | 59.5451 | 3.0172 |
| RF | 21.971 | 34.9809 |
| XGB | 24.0205 | 4.1868 |
학습 시간은 scaler+fit입니다. 추론은 이미 계산한 25D 특징 한 개의 scaler+predict이며,
예열 10회 후 100회 평균입니다. 신호 취득, 필터, 특징 추출 및 데이터 다운로드는 제외합니다.
환경은 results/environment.json을 참고합니다. Colab의 시간은 장치와 실행 상황에 따라 달라집니다.

## 3. Confusion Matrix 분석
![세 모델 혼동행렬](figures/confusion8.png)
행=실제, 열=예측입니다. error_pairs.csv, class_recall.csv, error_types.csv에 수치를 저장했습니다.

### SVM
![SVM 혼동행렬](figures/confusion_SVM.png)
- 가장 잘 분류된 클래스(재현율 기준, 공동 포함): 30_fist
- 재현율이 가장 낮은 클래스(공동 포함): 60_fist
- 주요 방향별 오류 쌍:
| Model | true | predicted | count | type |
| --- | --- | --- | --- | --- |
| SVM | 60_supination | 30_supination | 160 | angle_only |
| SVM | 30_supination | 60_supination | 151 | angle_only |
| SVM | 30_pronation | 30_supination | 150 | gesture_only |
| SVM | 60_pronation | 30_pronation | 142 | angle_only |
| SVM | 60_open | 30_open | 141 | angle_only |

[직접 작성] 각도만 틀린 오류와 동작만 틀린 오류 중 어느 쪽이 많은지 수치로 설명합니다.
[직접 작성] 가능한 원인과 추가 확인 방법을 적습니다. 혼동행렬만으로 근육이나 생리적 원인을 확정하지 않습니다.

### RF
![RF 혼동행렬](figures/confusion_RF.png)
- 가장 잘 분류된 클래스(재현율 기준, 공동 포함): 30_fist
- 재현율이 가장 낮은 클래스(공동 포함): 60_fist
- 주요 방향별 오류 쌍:
| Model | true | predicted | count | type |
| --- | --- | --- | --- | --- |
| RF | 60_open | 30_open | 122 | angle_only |
| RF | 30_fist | 60_fist | 91 | angle_only |
| RF | 30_supination | 30_pronation | 90 | gesture_only |
| RF | 60_supination | 30_supination | 87 | angle_only |
| RF | 60_fist | 30_fist | 86 | angle_only |

[직접 작성] 각도만 틀린 오류와 동작만 틀린 오류 중 어느 쪽이 많은지 수치로 설명합니다.
[직접 작성] 가능한 원인과 추가 확인 방법을 적습니다. 혼동행렬만으로 근육이나 생리적 원인을 확정하지 않습니다.

### XGB
![XGB 혼동행렬](figures/confusion_XGB.png)
- 가장 잘 분류된 클래스(재현율 기준, 공동 포함): 30_open
- 재현율이 가장 낮은 클래스(공동 포함): 60_fist
- 주요 방향별 오류 쌍:
| Model | true | predicted | count | type |
| --- | --- | --- | --- | --- |
| XGB | 30_fist | 60_fist | 116 | angle_only |
| XGB | 60_open | 30_open | 115 | angle_only |
| XGB | 60_pronation | 30_pronation | 99 | angle_only |
| XGB | 30_pronation | 60_pronation | 97 | angle_only |
| XGB | 30_open | 60_open | 90 | angle_only |

[직접 작성] 각도만 틀린 오류와 동작만 틀린 오류 중 어느 쪽이 많은지 수치로 설명합니다.
[직접 작성] 가능한 원인과 추가 확인 방법을 적습니다. 혼동행렬만으로 근육이나 생리적 원인을 확정하지 않습니다.

### 추가 분석
| Model | window_accuracy | trial_vote_accuracy | tie_trials |
| --- | --- | --- | --- |
| SVM | 0.4782 | 0.7448 | 25 |
| RF | 0.6527 | 0.9714 | 3 |
| XGB | 0.6358 | 0.9531 | 5 |
윈도우 단위와 시행 다수결 정확도를 비교합니다. 다수결 동률은 인코딩 순서로 처리했습니다.
[직접 작성] 집계 단위가 바뀌면 왜 성능이 달라지는지, 높은 다수결 성능의 실제 의미가 무엇인지 설명합니다.

## 4. 최종 결과
- 이번 CV 평균 정확도가 가장 높은 모델: RF
- [직접 작성] 가장 낮은 모델, 주요 오류 클래스, 정확도와 비용을 함께 고려한 선택
- [직접 작성] 실험 결과로 알게 된 점과 아직 확인하지 못한 점

### 한계
1. Trial-wise 분할이므로 같은 사람/같은 기록 파일의 다른 시행이 학습과 평가에 함께 들어갈 수 있고, 새 피험자 일반화는 LOSO 등 별도 검증이 필요합니다.
2. mark 기준의 분석 시작점 +1.5초는 가정이며 실제 근활성 구간 정렬을 확정하지 않았습니다.
3. 제공 데이터는 이미 필터링되어 있으며 이번 실습은 추가 필터를 적용하였습니다. 논문의 처리와 정확히 동일하다고 주장하지 않습니다.
4. 19개 겹친 윈도우는 독립된 19회 실험이 아니며, 윈도우 정확도와 시행 정확도를 구분합니다.
5. 5-fold p값은 작은 표본과 폴드 간 의존성의 영향을 받습니다. 유의하지 않음은 동등성의 증거가 아닙니다.
6. Colab 계산 비용은 특정 실행 환경의 결과이며, 임베디드 실시간 시스템의 전체 지연을 뜻하지 않습니다.

## 재현 정보
분할/모델 seed=42, CV=5-fold stratified, 주요 설정은 results/config.json 및 model_parameters.json,
환경 버전은 results/environment.json, 분할 목록은 manifest 파일에 기록했습니다.
