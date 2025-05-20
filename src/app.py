from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import numpy as np
import json
from datetime import datetime, date
import os

app = FastAPI(
    title="블록체인 데이터 시각화 API",
    description="블록체인 트랜잭션 분석 및 시각화를 위한 API",
    version="1.0.0"
)

# CORS 미들웨어 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 오리진 허용 (프로덕션에서는 특정 도메인으로 제한하는 것이 좋음)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 요청 모델 정의
class NodeRequest(BaseModel):
    start_date: Optional[str] = "2021-02-01"  # "YYYY-MM-DD" 형식
    end_date: Optional[str] = "2021-02-22"    # "YYYY-MM-DD" 형식
    batch_quant_weight: float = 50.0
    tx_count_weight: float = 50.0
    tx_amount_weight: float = 50.0
    top_n: int = 1

# 파일 경로 설정
DATA_DIR = "data"
ORIGINAL_DATA_PATH = os.path.join("..", DATA_DIR, "transfers_1643003325000_1643097085000.csv")

# 데이터 로드 함수
def load_data():
    """원본 CSV 데이터 로드"""
    try:
        df = pd.read_csv(ORIGINAL_DATA_PATH, encoding='utf-8')
        # 타임스탬프를 날짜 형식으로 변환
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms').dt.date
        return df
    except Exception as e:
        print(f"데이터 로드 오류: {e}")
        return pd.DataFrame()

# 파생 데이터 계산 함수
def calculate_derived_features(df):
    """datamining.py를 참고하여 파생 특성 계산"""
    try:
        # IBCReceive 제거
        df = df[df['type'] != 'IBCReceive'].copy()

        # 외부 체인 거래 여부
        df['is_external'] = df['fromChain'] != df['toChain']

        # 보낸 트랜잭션 집계
        sent_stats = (
            df.groupby('fromAddress')
            .agg(sent_tx_count=('amount', 'count'), sent_tx_amount=('amount', 'sum'))
            .rename_axis('address')
        )

        # 받은 트랜잭션 집계
        recv_stats = (
            df.groupby('toAddress')
            .agg(recv_tx_count=('amount', 'count'), recv_tx_amount=('amount', 'sum'))
            .rename_axis('address')
        )

        # 외부체인 보낸 트랜잭션 집계
        external_df = df[df['is_external']]
        external_sent_stats = (
            external_df.groupby('fromAddress')
            .agg(external_sent_tx_count=('amount', 'count'), external_sent_tx_amount=('amount', 'sum'))
            .rename_axis('address')
        )

        # 외부체인 받은 트랜잭션 집계
        external_recv_stats = (
            external_df.groupby('toAddress')
            .agg(external_recv_tx_count=('amount', 'count'), external_recv_tx_amount=('amount', 'sum'))
            .rename_axis('address')
        )

        # 모든 통계 합치기
        total_stats = (
            sent_stats
            .join(recv_stats, how='outer')
            .join(external_sent_stats, how='outer')
            .join(external_recv_stats, how='outer')
        )

        # 시간대별 Shannon entropy 계산
        temp_df = df.copy()
        temp_df['timestamp'] = pd.to_datetime(temp_df['timestamp'], unit='ms')
        temp_df['hour'] = temp_df['timestamp'].dt.hour

        sent = temp_df[['fromAddress', 'hour']].rename(columns={'fromAddress': 'address'})
        hourly = (
            sent.groupby(['address', 'hour'])
            .size()
            .rename('hour_count')
            .reset_index()
        )

        def shannon_entropy(counts):
            p = counts / counts.sum()
            return -(p * np.log2(p)).sum() if len(counts) > 0 else 0

        ent = (
            hourly.groupby('address')['hour_count']
            .apply(shannon_entropy)
            .rename('hour_entropy')
            .reset_index()
        )

        total_stats = (
            total_stats.reset_index()
            .merge(ent, on='address', how='left')
            .fillna(0)
            .set_index('address')
        )

        # 활동 일수 기반 지표 추가
        date_stats = (
            temp_df.groupby('fromAddress')['timestamp']
            .agg(
                first_date=lambda x: x.min().strftime('%Y-%m-%d') if not x.empty else None,
                last_date=lambda x: x.max().strftime('%Y-%m-%d') if not x.empty else None,
                active_days_count=lambda x: x.dt.date.nunique() if not x.empty else 0
            )
            .rename_axis('address')
            .reset_index()
        )

        total_stats = (
            total_stats.reset_index()
            .merge(date_stats, on='address', how='left')
            .set_index('address')
        )
        total_stats['active_days_count'] = total_stats['active_days_count'].fillna(0)

        # 거래 상대방 다양성 추가
        cp_count_sent = (
            df.groupby('fromAddress')['toAddress']
            .nunique()
            .rename('counterparty_count_sent')
            .reset_index()
            .rename(columns={'fromAddress': 'address'})
        )

        cp_count_recv = (
            df.groupby('toAddress')['fromAddress']
            .nunique()
            .rename('counterparty_count_recv')
            .reset_index()
            .rename(columns={'toAddress': 'address'})
        )

        total_stats = (
            total_stats.reset_index()
            .merge(cp_count_sent, on='address', how='left')
            .merge(cp_count_recv, on='address', how='left')
            .set_index('address')
        )

        total_stats['counterparty_count_sent'] = total_stats['counterparty_count_sent'].fillna(0)
        total_stats['counterparty_count_recv'] = total_stats['counterparty_count_recv'].fillna(0)

        # 거래금액 특성 추가
        total_stats['sent_tx_amount_mean'] = total_stats['sent_tx_amount'] / total_stats['sent_tx_count'].replace(0, np.nan)
        total_stats['recv_tx_amount_mean'] = total_stats['recv_tx_amount'] / total_stats['recv_tx_count'].replace(0, np.nan)
        total_stats['external_sent_tx_amount_mean'] = total_stats['external_sent_tx_amount'] / total_stats['external_sent_tx_count'].replace(0, np.nan)
        total_stats['external_recv_tx_amount_mean'] = total_stats['external_recv_tx_amount'] / total_stats['external_recv_tx_count'].replace(0, np.nan)

        # NaN 제거
        total_stats.fillna(0, inplace=True)

        return total_stats.reset_index()
    
    except Exception as e:
        print(f"파생 특성 계산 오류: {e}")
        return pd.DataFrame()

# 가중치 적용 및 상위 노드 선별 함수
def apply_weights_and_get_top_nodes(derived_df, batch_quant_weight, tx_count_weight, tx_amount_weight, top_n=10):
    """가중치를 적용하여 상위 노드 선별"""
    try:
        # 데이터프레임 복사
        df = derived_df.copy()
        
        # 정규화할 특성 선택
        features = ['hour_entropy', 'active_days_count', 'sent_tx_count', 'recv_tx_count', 
                   'sent_tx_amount', 'recv_tx_amount', 'counterparty_count_sent', 'counterparty_count_recv']
        
        # 정규화 (Min-Max 스케일링)
        for feature in features:
            if feature in df.columns:
                min_val = df[feature].min()
                max_val = df[feature].max()
                if max_val > min_val:  # 0으로 나누기 방지
                    df[f'{feature}_norm'] = (df[feature] - min_val) / (max_val - min_val)
                else:
                    df[f'{feature}_norm'] = 0
        
        # 배치/퀀트 점수 계산 (낮은 엔트로피 = 높은 규칙성 = 높은 배치/퀀트 특성)
        df['batch_quant_score'] = (
            (1 - df['hour_entropy_norm']) * 0.5 + 
            df['active_days_count_norm'] * 0.3 +
            (df['sent_tx_count_norm'] / (df['counterparty_count_sent_norm'] + 0.001)) * 0.2
        )
        
        # 거래 횟수 점수
        df['tx_count_score'] = (
            df['sent_tx_count_norm'] * 0.5 + 
            df['recv_tx_count_norm'] * 0.5
        )
        
        # 거래량 점수
        df['tx_amount_score'] = (
            df['sent_tx_amount_norm'] * 0.5 + 
            df['recv_tx_amount_norm'] * 0.5
        )
        
        # 가중치 정규화
        total_weight = batch_quant_weight + tx_count_weight + tx_amount_weight
        if total_weight > 0:  # 0으로 나누기 방지
            norm_batch_quant_weight = batch_quant_weight / total_weight
            norm_tx_count_weight = tx_count_weight / total_weight
            norm_tx_amount_weight = tx_amount_weight / total_weight
        else:
            norm_batch_quant_weight = norm_tx_count_weight = norm_tx_amount_weight = 1/3
        
        # 최종 점수 계산
        df['final_score'] = (
            df['batch_quant_score'] * norm_batch_quant_weight +
            df['tx_count_score'] * norm_tx_count_weight +
            df['tx_amount_score'] * norm_tx_amount_weight
        )
        
        # 티어 할당
        df['tier'] = pd.cut(
            df['final_score'], 
            bins=[0, 0.25, 0.5, 0.75, 1], 
            labels=['브론즈', '실버', '골드', '다이아몬드'],
            include_lowest=True
        )
        
        # 체인 정보 추출 (주소에서 첫 점까지)
        df['chain'] = df['address'].str.split('.').str[0]
        df['chain'] = df['chain'].str.split('1').str[0]
        
        # 상위 N개 노드 선택
        top_nodes = df.sort_values('final_score', ascending=False).head(top_n)
        
        return top_nodes
    
    except Exception as e:
        print(f"가중치 적용 오류: {e}")
        return pd.DataFrame()

# 관련 노드 추출 함수
def get_related_nodes(original_df, top_node_addresses):
    """상위 노드와 1-depth 관계에 있는 노드 추출"""
    try:
        # 상위 노드가 발신자인 거래에서 수신자 추출
        sent_related = original_df[original_df['fromAddress'].isin(top_node_addresses)]['toAddress'].unique()
        
        # 상위 노드가 수신자인 거래에서 발신자 추출
        recv_related = original_df[original_df['toAddress'].isin(top_node_addresses)]['fromAddress'].unique()
        
        # 관련 노드 합치기 (상위 노드는 제외)
        related_nodes = set(sent_related).union(set(recv_related)).difference(set(top_node_addresses))
        
        return list(related_nodes)
    
    except Exception as e:
        print(f"관련 노드 추출 오류: {e}")
        return []

# API 엔드포인트: 노드 데이터 가져오기
@app.post("/get_nodes")
async def get_nodes(request: NodeRequest):
    try:
        # 1. 날짜 범위에 맞는 데이터 추출
        original_df = load_data()
        
        # 날짜 필터링
        if request.start_date:
            start_date = datetime.strptime(request.start_date, "%Y-%m-%d").date()
            original_df = original_df[original_df['date'] >= start_date]
        
        if request.end_date:
            end_date = datetime.strptime(request.end_date, "%Y-%m-%d").date()
            original_df = original_df[original_df['date'] <= end_date]
        
        # 데이터가 없는 경우
        if original_df.empty:
            return {
                "top_nodes_json": [],
                "related_nodes_json": [],
                "top_nodes_derived_json": [],
                "related_nodes_derived_json": []
            }
        
        # 2. 파생 데이터 계산
        derived_df = calculate_derived_features(original_df)
        print('@@@ 2', derived_df)
        
        # 3. 가중치 적용하여 상위 노드 추출
        top_nodes_derived = apply_weights_and_get_top_nodes(
            derived_df,
            request.batch_quant_weight,
            request.tx_count_weight,
            request.tx_amount_weight,
            request.top_n
        )
        print('@@@ 3', top_nodes_derived)
        
        # 4. 상위 노드 주소 목록 추출
        top_node_addresses = top_nodes_derived['address'].tolist()
        print('@@@ 4', top_nodes_derived)
        
        # 5. 관련 노드 추출 (1-depth)
        related_node_addresses = get_related_nodes(original_df, top_node_addresses)
        print('@@@ 5', related_node_addresses)
        
        # 6. 원본 데이터에서 상위 노드와 관련 노드 데이터 추출
        top_nodes = original_df[
            (original_df['fromAddress'].isin(top_node_addresses)) | 
            (original_df['toAddress'].isin(top_node_addresses))
        ]
        
        related_nodes = original_df[
            (original_df['fromAddress'].isin(related_node_addresses)) | 
            (original_df['toAddress'].isin(related_node_addresses))
        ]
        print('@@@ 6', 'top_nodes', top_nodes)
        print('@@@ 6', 'top_nodes_derived', top_nodes_derived)
        
        # 7. top_nodes_derived에서 상위 노드 데이터 추출
        top_nodes_derived_ret = derived_df[derived_df['address'].isin(top_node_addresses)]
        print('@@@ 7', top_nodes_derived_ret)

        # 8. 관련 노드의 파생 데이터 계산
        related_nodes_derived = derived_df[derived_df['address'].isin(related_node_addresses)]
        print('@@@ 8', related_nodes_derived)

        # 9-12. JSON 변환
        top_nodes_json = json.loads(top_nodes.to_json(orient='records'))
        related_nodes_json = json.loads(related_nodes.to_json(orient='records'))
        top_nodes_derived_json = json.loads(top_nodes_derived_ret.to_json(orient='records'))
        related_nodes_derived_json = json.loads(related_nodes_derived.to_json(orient='records'))
        print('@@@ 9-1', len(top_nodes_json))
        print('@@@ 9-2', len(related_nodes_json))
        print('@@@ 9-3', len(top_nodes_derived_json))
        print('@@@ 9-4', len(related_nodes_derived_json))

        # 13. 결과 반환
        return {
            "top_nodes_json": top_nodes_json,
            "related_nodes_json": related_nodes_json,
            "top_nodes_derived_json": top_nodes_derived_json,
            "related_nodes_derived_json": related_nodes_derived_json
        }
    
    except Exception as e:
        print(f"API 처리 오류: {e}")
        # 에러가 발생해도 동일한 형식 유지
        return {
            "top_nodes_json": [],
            "related_nodes_json": [],
            "top_nodes_derived_json": [],
            "related_nodes_derived_json": []
        }

# 서버 구동
if __name__ == "__main__":
    import uvicorn
    
    # 데이터 디렉토리 확인
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"'{DATA_DIR}' 디렉토리가 생성되었습니다. 데이터 파일을 넣어주세요.")
    
    # 서버 시작
    uvicorn.run(app, host="0.0.0.0", port=8000)
