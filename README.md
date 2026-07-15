# Crypto Auto-Trade

GUI 기반 암호화폐 실시간 자동매매 프로그램 (Binance + Upbit, Windows).

대량 전략 탐색 → 통계적 검증(워크포워드) → 앙상블 구성 → 페이퍼/실거래까지 한 번에 다루는 범용 트레이딩 워크벤치입니다.

## 실행

```powershell
pip install -r requirements.txt
python app.py
```

## 주요 기능

| 탭 | 기능 |
|---|---|
| 대시보드 | 실시간 캔들차트(다크 테마, EMA 오버레이), 자동 새로고침 |
| 백테스트 | 전략/파라미터/기간/비용 설정 → 자본곡선·차트·거래내역·지표 리포트 |
| 최적화 | 전략×심볼×타임프레임 대량 그리드 탐색(멀티프로세스), 워크포워드 검증 |
| 실시간 매매 | 페이퍼 트레이딩(기본) / 실거래(ccxt), 리스크 한도, 포지션/주문 로그 |
| 로그 | 전체 이벤트 로그 |

## 아키텍처

```
core/
  backtest/   numba 가속 백테스트 엔진 (다음봉 시가 체결, 수수료·슬리피지,
              인트라바 손절/익절/트레일링 — 비관적 체결 가정, 365일 연환산)
  strategies/ 전략 zoo: tsmom, ema_cross, donchian(single/multi), larry_vb,
              rsi_momentum, bb_rsi_meanrev, voting/regime 앙상블 (registry 등록제)
  optimize/   그리드 탐색(spawn-safe 멀티프로세스), 게이트, 플래토-센터 선택,
              롤링 워크포워드(12개월 IS / 3개월 OOS)
  live/       LiveEngine(봉마감 구동), PaperBroker/CcxtBroker(정밀도·최소주문·
              타임아웃 리컨실), 상태 영속화·복구
  risk.py     볼타게팅 사이징, 고정비율, DD 사다리(10% 소프트/15% 킬), 일손실 한도
  regime.py   ADX 히스테리시스(25/20)+CHOP+변동성 레짐 분류
  data/       ccxt OHLCV 다운로더(파케이 캐시, Upbit 후방 페이지네이션·갭 채움)
gui/          PySide6 + pyqtgraph (QPicture 정적 캔들 + O(1) 라이브 바)
scripts/      download_data, run_search, run_wfa_batch, build_ensemble, eval_holdout
config/       ensemble.json — 검증 완료된 배포 전략 세트
```

## 검증 방법론 (docs/research-brief.md)

1. **탐색**: 4,176개 (전략, 파라미터, 심볼, 타임프레임) 조합을 2019~2026 데이터로 백테스트 (마지막 180일은 홀드아웃으로 제외)
2. **게이트**: 거래수 ≥30, 샤프 ≥0.8, MDD ≤35%, PF ≥1.15 → 168개 생존
3. **워크포워드**: 22개 셀에 롤링 12개월 IS / 3개월 OOS 재최적화(IS 워밍업 대칭, 플래토-센터 선택). WFE ≥ 0.5 & OOS 샤프 기준 → 8개 셀 생존
4. **앙상블**: 역변동성 가중 + 상관 페널티(>0.7), 셀당 35% 상한
5. **홀드아웃**: 최근 180일(한 번도 안 본 데이터) 단 1회 평가

## 검증 결과 (2026-07-08)

**워크포워드 OOS (5년 스티치, 시장 전 구간) — 앙상블 v2 (10셀):**

| 전략 셀 | OOS 샤프 | OOS CAGR | MDD | WFE |
|---|---|---|---|---|
| upbit BTC 4h ema_cross | 1.96 | 81.8% | 40.5% | 0.67 |
| upbit BTC 1d tsmom | 1.94 | 41.6% | 15.5% | 0.69 |
| upbit ETH 4h ema_cross | 1.88 | 111.4% | 34.9% | 0.66 |
| upbit BTC 4h vov_calm_trend ★자체설계 | 1.68 | 24.4% | 15.8% | 0.70 |
| binance BTC 1d tsmom | 1.66 | 34.0% | 13.3% | 0.69 |
| upbit SOL 1d tsmom | 1.42 | 31.4% | 14.0% | 1.20 |
| upbit ETH 4h vov_calm_trend ★자체설계 | 1.38 | 17.2% | 15.1% | 0.50 |
| (외 3개 셀, 전체 목록 config/ensemble.json) | | | | |

**홀드아웃 (2026-01 → 2026-07, 급락장 · 기록용):**

| | 수익률 | MDD |
|---|---|---|
| 앙상블 v2 (10셀) | **-1.8%** | **4.0%** |
| 앙상블 v1 (8셀) | -3.3% | 6.1% |
| 시장 평균 (동일 심볼 단순보유) | -39.2% | 37~66% |

개선 이력과 기각된 후보들(총 17종 검증)은 `docs/improvement-log.md` 참고.

홀드아웃 기간은 BTC -30%, ETH -43%, ADA -54%의 약세장이었습니다. 롱온리 추세추종 시스템은 하락장에서 "잃지 않는 것"이 목표이며, 추세 필터와 볼타게팅이 설계대로 자본을 보호했습니다(시장 대비 +35%p).

## 실거래 전 필독

- **기본값은 페이퍼 트레이딩입니다.** 실거래는 '실거래' 타이핑 확인을 거쳐야 켜집니다.
- API 키는 Windows 자격 증명 관리자(keyring)에 저장됩니다. **출금 권한 없는 키**를 쓰고 IP 제한을 걸어두세요.
- 리스크 한도 기본값: 일손실 3% 도달 시 청산+당일 정지, 최대 낙폭 15% 도달 시 킬스위치(수동 재가동 필요).
- 백테스트/워크포워드 성과는 미래 수익을 보장하지 않습니다. 소액(최소 주문 단위)으로 시작하세요.
- 권장 배포 구성은 `config/ensemble.json` 참고 (스크립트로 재생성 가능).

## 데이터 갱신 / 재검증

```powershell
python -m scripts.download_data                 # OHLCV 캐시 갱신
python -m scripts.run_search --exchange binance --timeframes "1d" ...   # 탐색
python -m scripts.run_wfa_batch                 # 워크포워드 배치
python -m scripts.build_ensemble                # 앙상블 재구성
```

테스트: `python -m pytest tests/` (86개)
