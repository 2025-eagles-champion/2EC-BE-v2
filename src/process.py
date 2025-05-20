import os
import pandas as pd
import glob
import os

# raw 폴더 내의 모든 CSV 파일 경로 가져오기

if not os.path.exists(os.path.join('..', 'raw')):
    os.mkdir(os.path.join('..', 'raw'))

_path = os.path.join('..', 'raw', '*.csv')
csv_files = glob.glob(_path)

for file_path in csv_files:
    print(f"처리 중인 파일: {file_path}")

    try:
        # CSV 파일 읽기
        df = pd.read_csv(file_path)

        # 처리 전 행 수 기록
        rows_before = len(df)

        # "IBCReceive" 타입 행 삭제
        df = df[df['type'] != 'IBCReceive']

        # 처리 후 행 수 기록
        rows_after = len(df)
        rows_removed = rows_before - rows_after

        # 변경된 데이터프레임을 원본 파일에 저장
        df.to_csv(file_path, index=False)

        print(f"파일 처리 완료: {file_path}")
        print(f"삭제된 행 수: {rows_removed}")

    except Exception as e:
        print(f"파일 처리 중 오류 발생: {file_path}")
        print(f"오류 내용: {e}")

print("모든 파일 처리가 완료되었습니다.")