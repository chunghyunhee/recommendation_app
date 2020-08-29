import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# 전처리 완료된 데이터
df_recom = pd.read_csv('https://github.com/TaewoongKong/code_sharing/blob/master/cau_place_final_data.csv?raw=True')

# 리뷰의 유사도 학습
count_vect_review = CountVectorizer(min_df=2, ngram_range=(1, 2))
place_review = count_vect_review.fit_transform(df_recom['review_txt'])
place_simi_review = cosine_similarity(place_review, place_review)
place_simi_review_sorted_ind = place_simi_review.argsort()[:, ::-1]

# 카테고리의 유사도 학습
count_vect_category = CountVectorizer(min_df=0, ngram_range=(1, 2))
place_category = count_vect_category.fit_transform(df_recom['cate_mix'])
place_simi_cate = cosine_similarity(place_category, place_category)
place_simi_cate_sorted_ind = place_simi_cate.argsort()[:, ::-1]

# 카테고리와 리뷰의 중요성을 짬뽕시키는 공식
place_simi_co = (place_simi_review * 0.4  # 리뷰 유사도 반영
                 + place_simi_cate * 0.2  # 카테고리 유사도 반영
                 + np.repeat([df_recom['star_qty'].values], len(df_recom['star_qty']), axis=0) * 0.0005  # 별점평가 개수
                 + np.repeat([df_recom['star_point'].values], len(df_recom['star_point']), axis=0) * 0.01  # 별점 평가 반영
                 + np.repeat([df_recom['sentiment'].values], len(df_recom['sentiment']), axis=0) * 0.005  # 감정분석 개수

                 )

# 필요한 작업
place_simi_co_sorted_ind = place_simi_co.argsort()[:, ::-1]


# 위경도 구하는 공식
def haversine_distance(lat1, lon1, lat2, lon2):
    r = 6371
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    a = np.sin(delta_phi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2) ** 2
    res = r * (2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))
    return np.round(res, 2)


# 추천엔진 돌리는 최종 함수
def find_simi_place(df, sorted_ind, place_name, top_n):
    place_title = df[df['name'] == place_name]
    place_index = place_title.index.values
    similar_indexes = sorted_ind[place_index, :(top_n + 1)]
    similar_indexes = similar_indexes.reshape(-1)
    result_df = df.iloc[similar_indexes]

    start_lat, start_lon = result_df['lat'].tolist()[0], result_df['lon'].tolist()[0]

    distances_km = []
    for row in result_df.itertuples(index=False):
        distances_km.append(haversine_distance(start_lat, start_lon, row.lat, row.lon))

    result_df['distance'] = distances_km

    return result_df.loc[~(result_df['name'] == place_name)].sort_values(by='distance')



## check final results
find_simi_place(df_recom,place_simi_co_sorted_ind, "사리원", 7)