# FinRL-X Trading Lab

새 프로젝트의 백엔드와 프론트엔드는 이 폴더에 있습니다. 예전 앱 폴더는 참조하지 않습니다.

## 구조

```text
FinRL-Trading feed / StrategyResult / BacktestEngine
             ↓ OHLCV와 causal feature
기존 champion 합체 모델의 선택적 frozen feature adapter
             ↓ asset feature + expert latent + account state
TorchRL EnvBase + Collector + 등록 가능한 learner
             ↓ 실종목 target weights (합계 ≤ 1)
FinRL StrategyResult / bt 현금 계정 / paper inference
```

TorchRL actor는 종목별 공유 encoder와 masked cross-asset pooling을 씁니다. 파라미터는 ticker 이름과 종목 개수에 종속되지 않으며, 배치에 padding이 있으면 `asset_mask`가 이를 제외합니다. 각 종목은 연속 target weight를 내고 남은 비중은 account cash입니다. 주문 실행, 수수료, 사용 가능한 가격은 환경과 execution 경계에서 처리합니다.

PPO는 learner baseline입니다. learner registry는 정책과 `EnvBase`에서 분리되어 있습니다. 과거 rollout, PPO 수집, 결정론적 실시간 paper 추론은 같은 actor 구조와 관측 계약을 사용합니다. 실시간 paper는 학습 Collector와 별도 시세 queue를 읽고 같은 actor의 deterministic helper로 행동합니다. 체크포인트 호환성은 architecture version, feature schema, Expert latent schema를 확인하며 ticker 목록을 저장하지 않습니다.

FinRL-Trading의 설치된 `BacktestEngine`은 공개 `run_backtest()`와 `_create_bt_strategy()` 안에서 종목 비중을 1로 정규화합니다. 프로젝트 adapter는 그 정규화 단계만 우회하고 설치 엔진의 가격 준비, 수수료 계산, 지표 계산과 `bt`의 `WeighTarget`을 재사용합니다. `bt`에는 실제 종목만 전달하고 목표 종목 비중 합계가 1보다 작으면 잔액은 `bt` 현금으로 남습니다. 종목 가중치가 전부 0이면 주문은 없습니다.

신호 시각과 실행 시각은 나눕니다. 정책은 `bar_close[t]`에서 상태를 읽고, 다음 이용 가능한 종가 `t+1`에서 목표를 실행합니다. live environment와 backtest 모두 실행 시점에 목표를 적용하고 같은 거래비용 0.1%를 씁니다. paper ledger에도 현금 잔액을 별도 보유하며 매도·매수 주문만 기록합니다.

## champion.pt 확인 결과와 연결 상태

설정된 `champion.pt`는 `registered_vertical_trading_moe_v1` 형식이며 20개 manifest와 3,570개 state tensor가 있습니다. `controller.market_fusion` 64차원 계층과 개별 Expert용 `native_runner_source` 문자열은 존재합니다. 체크포인트에는 가중치와 개별 runner는 있지만 합체 Controller의 forward 구현과 실행 가능한 통합 adapter가 없습니다. state dict만으로 실행 순서나 mask/router 의미를 추측해 만든 가짜 추론은 넣지 않았습니다.

따라서 champion manifest는 검사하지만, 실제 latent 생성은 `inference_ready=false`로 나타납니다. live bar에 검증된 64차원 latent가 붙을 때만 actor가 이를 사용하고, 현재는 `expert_mask=false`로 관측합니다. 역사 latent cache는 forward adapter가 있을 때 한 번 batch 계산해 저장하며 rollout 중 Expert를 재실행하지 않습니다. 현재 adapter가 없으므로 허위 latent를 만들거나 저장하지 않습니다. 체크포인트 안의 MSFT PPO actor는 새 shared allocation policy와 별도 역할이므로 새로운 RL 정책이나 expert fusion 출력으로 쓰지 않습니다.

정책은 연속적인 비중과 현금 잔액을 냅니다. BUY/HOLD/SELL 규칙, 균등배분, ticker별 상한, technical-score 매수 fallback은 없습니다. 실계좌 broker 실행은 아직 연결되지 않았고 live 주문 gate는 닫혀 있습니다.

## 화면과 실행

React + Vite + Tailwind CSS 화면은 backend REST/WebSocket을 사용해 실시간 시세, 체크포인트 연결 가능 여부, 학습 상태, 종목 목표 비중과 현금 잔액, FinRL-Trading 백테스트, paper 계좌를 표시합니다. 실행 파일은 `실행.cmd`와 `start.cmd`입니다. 설치는 `install.cmd`에서 준비합니다.

모델은 별도 폴더에서 읽습니다.

- `C:\Users\hushm\Desktop\모델\champion.pt`
- `C:\Users\hushm\Desktop\모델\experts\...`

경로는 `.env.example`의 `FINRLX_MODEL_DIR`, `FINRLX_CHAMPION_PATH`, `FINRLX_EXPERTS_DIR`로 바꿀 수 있습니다. FinRL-Trading 설치 버전은 2.0.2이며, 프로젝트 의존성은 공식 저장소 확인 commit `4409abe925c904e570be78ebfb5e77ac3491dff8`에 고정돼 있습니다.
