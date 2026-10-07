# FinRL-X Trading Lab

새 프로젝트의 백엔드와 프론트엔드는 이 폴더에 있습니다. 예전 앱 폴더는 참조하지 않습니다.

## 구조

```text
FinRL-X DataSourceManager + SQLite price store
             ↓ Yahoo adapter를 FinRL-X OHLCV schema로 적재
FinRL-X BaseStrategy / StrategyResult
             ↑ TorchRL shared allocation policy가 target weights 제공
FinRL-X BacktestEngine의 cash-preserving adapter
FinRL AlpacaManager paper 실행 adapter (실계좌 주문 게이트 닫힘)
```

FinRL-X 데이터 관리자가 가격 조회와 SQLite 캐시를 소유합니다. FMP 키 또는 캐시 데이터가 없을 때 Yahoo adapter가 데이터를 가져와 FinRL-X 표준 OHLCV 열로 바꾼 뒤 같은 SQLite 저장소에 적재합니다. 실시간 bar streaming은 FinRL-X 일봉 데이터 인터페이스에 없는 기능이라 provider adapter로 연결합니다.

TorchRL actor는 FinRL-X `BaseStrategy`의 target-weight 생성부에 들어가 `StrategyResult(gvkey, weight)`를 반환합니다. 종목별 공유 encoder와 masked cross-asset pooling을 쓰며 파라미터는 ticker 이름과 종목 수에 종속되지 않습니다. 각 종목의 연속 비중 합계가 1보다 작으면 나머지는 계좌 현금입니다.

PPO는 learner baseline입니다. learner registry는 정책과 `EnvBase`에서 분리되어 있습니다. 과거 rollout, PPO 수집, 결정론적 실시간 paper 추론은 같은 actor 구조와 관측 계약을 사용합니다. 실시간 paper는 학습 Collector와 별도 시세 queue를 읽고 같은 actor의 deterministic helper로 행동합니다. 체크포인트 호환성은 architecture version, feature schema, Expert latent schema를 확인하며 ticker 목록을 저장하지 않습니다.

FinRL-Trading 2.0.2의 공개 `BacktestEngine.run_backtest()`는 종목 비중 합계를 1로 정규화해 현금 비중을 없앱니다. 그래서 백테스트 adapter가 이 공개 메서드의 정규화 단계는 건너뛰고, 설치된 엔진의 가격 준비·수수료·지표 계산과 `bt`의 `WeighTarget`을 재사용합니다. 이 adapter는 해당 버전에 고정돼 private engine helper를 호출합니다. `bt`에는 실제 종목만 전달합니다. 비중 합계가 1보다 작으면 차액은 `bt`의 현금으로 남고, 모든 종목 비중이 0이면 현금 100%입니다.

신호 시각과 실행 시각은 나눕니다. 정책은 `bar_close[t]`에서 상태를 읽고, 다음 이용 가능한 종가 `t+1`에서 목표를 실행합니다. live environment와 backtest 모두 실행 시점에 목표를 적용하고 같은 거래비용 0.1%를 씁니다. paper ledger에도 현금 잔액을 별도 보유하며 매도·매수 주문만 기록합니다.

## champion.pt 확인 결과와 연결 상태

설정된 `champion.pt`는 `registered_vertical_trading_moe_v1` 형식이며 20개 manifest와 3,570개 state tensor가 있습니다. `controller.market_fusion` 64차원 계층과 개별 Expert용 `native_runner_source` 문자열은 존재합니다. 체크포인트에는 가중치와 개별 runner는 있지만 합체 Controller의 forward 구현과 실행 가능한 통합 adapter가 없습니다. state dict만으로 실행 순서나 mask/router 의미를 추측해 만든 가짜 추론은 넣지 않았습니다.

따라서 champion manifest는 검사하지만, 실제 latent 생성은 `inference_ready=false`로 나타납니다. live bar에 검증된 64차원 latent가 붙을 때만 actor가 이를 사용하고, 현재는 `expert_mask=false`로 관측합니다. 역사 latent cache는 forward adapter가 있을 때 한 번 batch 계산해 저장하며 rollout 중 Expert를 재실행하지 않습니다. 현재 adapter가 없으므로 허위 latent를 만들거나 저장하지 않습니다. 체크포인트 안의 MSFT PPO actor는 새 shared allocation policy와 별도 역할이므로 새로운 RL 정책이나 expert fusion 출력으로 쓰지 않습니다.

정책은 연속적인 종목 비중과 현금 잔액을 냅니다. BUY/HOLD/SELL 규칙, 균등배분, ticker별 상한, technical-score 매수 fallback은 없습니다. 기본 실행 모드는 로컬 paper ledger입니다. `.env`에서 `FINRLX_PAPER_EXECUTOR=alpaca_paper`와 FinRL-Trading 표준 `APCA_*` paper 계정 설정을 넣으면 공식 `AlpacaManager.execute_portfolio_rebalance()`를 paper 계정에 호출합니다. 이 adapter는 paper API URL만 허용하며, 실계좌 주문 게이트는 닫혀 있습니다.

## 현재 실행 확인

FinRL-Trading 데이터 관리자에서 AAPL/NVDA/AMD의 66개 일봉 row를 조회했고, FinRL `StrategyResult`에 AAPL 20%, NVDA 15%, 잔여 현금 65%가 전달되는 것을 확인했습니다. 백테스트 adapter에는 별도로 현금 100%와 주식 35% 신호를 넣어 실행했습니다. 현금 100%는 자산가치가 변하지 않았고, 부분 투자 신호는 남은 현금을 유지하면서 가격 변동을 반영했습니다. API health와 React 페이지도 로컬에서 확인했습니다.

현재 `runtime/policy_torchrl.pt`에 호환 가능한 학습 정책이 없어 실제 actor 기반 backtest/inference는 아직 실행할 수 없습니다. 기존 `champion.pt`도 manifest와 tensor는 읽히지만 합체 모델의 실행 가능한 forward adapter가 없어 Expert latent 추론은 연결되지 않은 상태입니다. 따라서 위 백테스트 확인은 FinRL cash adapter 경로 검증이며 학습 정책 성과 검증은 아닙니다. Yahoo polling은 지연 시세이며, Alpaca paper executor는 계정 자격증명이 없어 호출 검증하지 않았습니다.

## 화면과 실행

React + Vite + Tailwind CSS 화면은 backend REST/WebSocket을 사용해 실시간 시세, 체크포인트 연결 가능 여부, 학습 상태, 종목 목표 비중과 현금 잔액, FinRL-Trading 백테스트, paper 계좌를 표시합니다. 실행 파일은 `실행.cmd`와 `start.cmd`입니다. 설치는 `install.cmd`에서 준비합니다.

모델은 별도 폴더에서 읽습니다.

- `C:\Users\hushm\Desktop\모델\champion.pt`
- `C:\Users\hushm\Desktop\모델\experts\...`

경로는 `.env.example`의 `FINRLX_MODEL_DIR`, `FINRLX_CHAMPION_PATH`, `FINRLX_EXPERTS_DIR`로 바꿀 수 있습니다. FinRL-Trading 설치 버전은 2.0.2이며, 프로젝트 의존성은 공식 저장소 확인 commit `4409abe925c904e570be78ebfb5e77ac3491dff8`에 고정돼 있습니다.
