import streamlit as st
import pandas as pd
import numpy as np
from sklearn import preprocessing
import pickle
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as pg
import plotly.figure_factory as pf
import plotly.colors as pc

#Thiết lập các thông tin cần thiết

df_org = pd.read_csv("./daxuly.csv")
df_org = df_org.rename(columns={'label': 'xeploai'})
df_y1 = pd.read_csv('./test_data_for_demo_year1.csv')
df_y2 = pd.read_csv('./test_data_for_demo_year2.csv')
df_y3 = pd.read_csv('./test_data_for_demo_year3.csv')
df_y35 = pd.read_csv('./test_data_for_demo_year35.csv')

def xettn(row):
    reasons = []

    if row['tong_drl'] < 50:
        reasons.append('Điểm rèn luyện < 50')
    if row['tienganh'] == '0':
        reasons.append('Chưa có ngoại ngữ')
    if row['dtb_toankhoa'] < 5:
        reasons.append('Điểm trung bình < 5')
    if row['sotc_tichluy'] < row['tc_yeucau']:
        reasons.append('Không đủ tín chỉ yêu cầu')

    if not reasons:
        reasons.append('Lí do khác')

    return ', '.join(reasons)

def lydorot(df_org):
    df_rot = df_org[df_org['xeploai'] == 'Rớt']
    drl = ['drlnam1', 'drlnam2','drlnam3', 'drlnam4', 'drlnam5', 'drlnam6', 'drl_3_5']
    df_rot[drl] = df_rot[drl].fillna(0, inplace = True)
    df_rot['tong_drl'] = df_rot[drl].sum(axis=1)
    df_rot['vi_pham'] = df_rot.apply(xettn, axis=1)

    violations_exploded = df_rot['vi_pham'].str.split(', ').explode()

    violation_counts = violations_exploded.value_counts(normalize=True) * 100

    violation_counts_df = pd.DataFrame(pd.DataFrame({
        'Lý do': violation_counts.index,
        'Phần trăm': violation_counts.values}))
    violation_counts_df['Formatted Values'] = [f"{value:.1f}%" for value in violation_counts_df['Phần trăm']]
    return violation_counts_df

predict_cols1 = ['namsinh', 'gioitinh', 'noisinh', 'khoa', 'hedt', 'khoahoc',
       'chuyennganh2', 'tinhtrang', 'diachi_tinhtp', 'diemnamhoc1', 'drlnam1',
       'dien_tt', 'diem_tt', 'sotc_rot1', 'tctichluy1', 'socc_tienganh',
       'tienganh', 'canhcao']
predict_cols2 = ['namsinh', 'gioitinh', 'noisinh', 'khoa', 'hedt', 'khoahoc',
       'chuyennganh2', 'tinhtrang', 'diachi_tinhtp', 'diemnamhoc1',
       'diemnamhoc2', 'drlnam1', 'drlnam2', 'dien_tt', 'diem_tt', 'sotc_rot1',
       'sotc_rot2', 'tctichluy1', 'tctichluy2', 'socc_tienganh', 'tienganh',
       'canhcao']
predict_cols3 = ['namsinh', 'gioitinh', 'noisinh', 'khoa', 'hedt', 'khoahoc',
       'chuyennganh2', 'tinhtrang', 'diachi_tinhtp', 'diemnamhoc1',
       'diemnamhoc2', 'diemnamhoc3', 'drlnam1', 'drlnam2', 'drlnam3',
       'dien_tt', 'diem_tt', 'sotc_rot1', 'sotc_rot2', 'sotc_rot3',
       'tctichluy1', 'tctichluy2', 'tctichluy3', 'socc_tienganh', 'tienganh',
       'canhcao']
predict_cols35 = ['namsinh', 'gioitinh', 'noisinh', 'khoa', 'hedt', 'khoahoc',
       'chuyennganh2', 'tinhtrang', 'diachi_tinhtp', 'diemnamhoc1',
       'diemnamhoc2', 'diemnamhoc3', 'diem_3_5', 'drlnam1', 'drlnam2',
       'drlnam3', 'drl_3_5', 'dien_tt', 'diem_tt', 'sotc_rot1', 'sotc_rot2',
       'sotc_rot3', 'rotmon_3_5', 'tctichluy1', 'tctichluy2', 'tctichluy3',
       'tc_tichluy_3_5', 'socc_tienganh', 'tienganh', 'canhcao']

with open("./model_cb_sm_y1.pkl", "rb") as file:
    model_y1 = pickle.load(file)
with open("./model_xg_ts_y2.pkl", "rb") as file:
    model_y2 = pickle.load(file)
with open("./model_xg_ts_y3.pkl", "rb") as file:
    model_y3 = pickle.load(file)
with open("./model_lgb_ts_y35.pkl", "rb") as file:
    model_y35 = pickle.load(file)
    
def predict(df, df_org, model, predict_col):
    df_pr = df[predict_col]
    cate_col = df_pr.select_dtypes(include=['object']).columns
    label_encoder = preprocessing.LabelEncoder()
    for u in cate_col:
        label_encoder.fit(df_org[u])
        df_pr[u] = label_encoder.transform(df_pr[u])
    df_pr['xeploai'] = model.predict(df_pr)
    df_pr['xeploai'] = df_pr['xeploai'].replace({
         0: 'Rớt',
         1: 'Trung bình',
         2: 'Trung bình khá',
         3: 'Khá',
         4: 'Giỏi',
         5: 'Xuất sắc'
    })
    return df_pr['xeploai']

needed_cols1 = ['mssv', 'gioitinh', 'lopsh', 'khoa', 'hedt', 'khoahoc', 'diemnamhoc1', 'drlnam1', 'sotc_rot1', 'tctichluy1', 'socc_tienganh', 'canhcao', 'xeploai']
needed_cols2 = ['mssv', 'gioitinh', 'lopsh', 'khoa', 'hedt', 'khoahoc', 'diemnamhoc1', 'diemnamhoc2', 'drlnam1', 'drlnam2', 'sotc_rot1', 'sotc_rot2', 'tctichluy1', 'tctichluy2', 'socc_tienganh', 'canhcao', 'xeploai']
needed_cols3 = ['mssv', 'gioitinh', 'lopsh', 'khoa', 'hedt', 'khoahoc', 'diemnamhoc1', 'diemnamhoc2', 'diemnamhoc3', 'drlnam1', 'drlnam2', 'drlnam3', 'sotc_rot1', 'sotc_rot2', 'sotc_rot3', 'tctichluy1', 'tctichluy2', 'tctichluy3', 'socc_tienganh', 'canhcao', 'xeploai']
needed_cols35 = ['mssv', 'gioitinh', 'lopsh', 'khoa', 'hedt', 'khoahoc', 'diemnamhoc1', 'diemnamhoc2', 'diemnamhoc3', 'diem_3_5', 'drlnam1', 'drlnam2', 'drlnam3', 'drl_3_5', 'sotc_rot1', 'sotc_rot2', 'sotc_rot3', 'rotmon_3_5', 'tctichluy1', 'tctichluy2', 'tctichluy3', 'tc_tichluy_3_5', 'socc_tienganh', 'canhcao', 'xeploai']

def make_bar_chart(df, x, y, hue, value_on_cols, title, x_title, y_title, title_position, palette):
    if value_on_cols != 0:
        fig = px.bar(df,
              x=x, 
              y=y,
              color = x,
              title=title, 
              color_discrete_sequence=palette,
              text=value_on_cols)
        fig.update_traces(textposition='outside')
    elif value_on_cols == 0:
        fig = px.bar(df,
              x=x, 
              y=y,
              color = x,
              title=title, 
              color_discrete_sequence=palette)
    if hue != 0:
        fig = px.bar(df,
              x=x, 
              y=y,
              color = hue,
              text=y,
              barmode='group',
              title=title, 
              color_discrete_sequence=palette)
        fig.update_traces(textposition='outside')
    
    fig.update_layout(
            yaxis_title=y_title,
            xaxis_title=x_title,
            uniformtext_minsize=10,
            uniformtext_mode='hide',
            yaxis=dict(range=[0, max(df[y]) * 1.1]),
            title_x=title_position
        )
    
    st.plotly_chart(fig)
    
# NỘI DUNG CHÍNH

st.title('Dự đoán xếp loại tốt nghiệp sinh viên trường Đại học Công nghệ Thông tin TP.HCM')
st.divider()

y_opt = None
overall_search_s = None
overall_khoa_search_s = None
target_search_s = None
overall_search = None
target_search = None

# I. Thống kê sinh viên đã hoàn thành khóa học

with st.sidebar:
    st.write('Mời bạn chọn các chức năng chính:')
    statistic_all = st.checkbox('Thống kê sinh viên đã hoàn thành chương trình học')
    
if statistic_all:
    with st.sidebar:
        st.write('Mời bạn chọn phạm vi tra cứu:')
        overall_search_s = st.checkbox('Toàn trường', key='overall_search_s')
        overall_khoa_search_s = st.checkbox('Theo khoa')
        target_search_s = st.checkbox('Sinh viên cụ thể', key='target_search_s')
        
if overall_search_s:
    overall_search_s_1 = st.checkbox('Xem danh sách sinh viên')
    overall_search_s_2 = st.checkbox('Biểu đồ kết quả xếp loại tốt nghiệp')
    overall_search_s_3 = st.checkbox('Biểu đồ phân bố lý do rớt tốt nghiệp của các sinh viên')
    overall_search_s_4 = st.checkbox('Biểu đồ phân bố loại tốt nghiệp theo khóa')
    st.divider()
    if overall_search_s_1:
        xl0 = st.selectbox('Chọn loại tốt nghiệp:', ['Tất cả', 'Rớt', 'Trung bình', 'Trung bình khá', 'Khá', 'Giỏi', 'Xuất sắc'], key='xl0')
        if xl0 != 'Tất cả':
            df_show_0 = df_org[df_org['xeploai'] == xl0]
            csv_os1 = df_show_0.to_csv(index=False)
        else:
            df_show_0 = df_org.copy()
            csv_os1 = df_show_0.to_csv(index=False)
        st.dataframe(df_show_0[needed_cols35])
        st.download_button(label='Tải xuống danh sách sinh viên', data=csv_os1, file_name='DS_sinh_vien.csv', mime='text/csv', key='dl_os1')
    if overall_search_s_2:
        df_os2 = df_org['xeploai'].value_counts().reset_index()
        df_os2.columns = ['Xếp loại', 'Số lượng']
        make_bar_chart(df_os2, 'Xếp loại', 'Số lượng', 0, 'Số lượng', 'Kết quả xếp loại tốt nghiệp của toàn trường', 'Xếp loại', 'Số lượng sinh viên', 0.2, pc.sequential.Emrld)
    if overall_search_s_3:
        df_os3 = lydorot(df_org)
        make_bar_chart(df_os3, 'Lý do', 'Phần trăm', 0, 'Formatted Values', 'Biểu đồ thể hiện phần trăm từng lý do rớt tốt nghiệp trên tổng dữ liệu', 'Lý do', 'Phần trăm (%)', 0.15, pc.sequential.Magenta)
    if overall_search_s_4:
        df_os4 = df_org.groupby(['khoahoc', 'xeploai']).size().reset_index(name='count')
        make_bar_chart(df_os4, 'khoahoc', 'count', 'xeploai', 0, 'Kết quả tốt nghiệp của các khóa sinh viên', 'Khóa học', 'Số lượng sinh viên', 0.3, pc.sequential.RdBu)
        
if overall_khoa_search_s:
    khoa_overall = st.selectbox('Mời bạn chọn khoa:', [u for u in list(np.append(np.array('Toàn khoa'), df_org['khoa'].unique()))], key='khoa_overall')
    if khoa_overall == 'Toàn khoa':
        st.write('Mời bạn chọn các chức năng sau:')
        oksa1 = st.checkbox('Xem danh sách sinh viên với xếp loại tốt nghiệp', key='oksa1')
        oksa2 = st.checkbox('Biểu đồ thống kê xếp loại tốt nghiệp', key='oksa2')
        oksa3 = st.checkbox('Biểu đồ so sánh điểm trung bình giữa các khoa', key='oksa3')
        oksa4 = st.checkbox('Biểu đồ so sánh xếp loại tốt nghiệp giữa các khoa', key='oksa4')
        st.divider()
        if oksa1:
            xloksa1 = st.selectbox('Chọn loại tốt nghiệp:', ['Tất cả', 'Rớt', 'Trung bình', 'Trung bình khá', 'Khá', 'Giỏi', 'Xuất sắc'], key='xloksa1')
            if xloksa1 != 'Tất cả':
                df_oksa1 = df_org[df_org['xeploai'] == xloksa1]
                csv_oksa1 = df_oksa1.to_csv(index=False)
            else:
                df_oksa1 = df_org.copy()
                csv_oksa1 = df_oksa1.to_csv(index=False)
            st.dataframe(df_oksa1[needed_cols35])
            st.download_button(label='Tải xuống danh sách sinh viên', data=csv_oksa1, file_name='DS_sinh_vien.csv', mime='text/csv')
        if oksa2:
            df_oksa2 = df_org['xeploai'].value_counts().reset_index()
            df_oksa2.columns = ['Xếp loại', 'Số lượng']
            make_bar_chart(df_oksa2, 'Xếp loại', 'Số lượng', 0, 'Số lượng', 'Kết quả xếp loại tốt nghiệp của tất cả các khoa', 'Xếp loại', 'Số lượng sinh viên', 0.2, pc.sequential.Brwnyl)
        if oksa3:
            df_oksa3 = df_org.groupby('khoa')['dtb_toankhoa'].mean().reset_index()

            df_oksa3 = df_oksa3.sort_values(by='dtb_toankhoa', ascending=False)
            df_oksa3['dtb_toankhoa'] = df_oksa3['dtb_toankhoa'].round(2)

            make_bar_chart(df_oksa3, 'khoa', 'dtb_toankhoa', 0, 'dtb_toankhoa', 'Kết quả học tập của mỗi khoa', 'Khoa', 'Điểm trung bình', 0.3, pc.sequential.Agsunset)
        if oksa4:
            xloksa2 = st.selectbox('Chọn loại tốt nghiệp:', ['Tất cả', 'Rớt', 'Trung bình', 'Trung bình khá', 'Khá', 'Giỏi', 'Xuất sắc'], key='xloksa2')
            if xloksa2 == 'Tất cả':
                df_oksa4 = df_org.groupby(['khoa', 'xeploai']).size().reset_index(name='count')
                make_bar_chart(df_oksa4, 'khoa', 'count', 'xeploai', 'count', 'Biểu đồ so sánh xếp loại tốt nghiệp của tất cả các khoa', 'Khoa', 'Số lượng sinh viên', 0.1, pc.sequential.RdBu)
            else:
                df_oksa4 = df_org.groupby(['khoa', 'xeploai']).size().reset_index(name='count')
                df_oksa4 = df_oksa4[df_oksa4['xeploai'] == xloksa2]
                if len(df_oksa4) != 0:
                    make_bar_chart(df_oksa4, 'khoa', 'count', 0, 'count', 'Biểu đồ so sánh xếp loại tốt nghiệp là ' + xloksa2 + ' của tất cả các khoa', 'Khoa', 'Số lượng sinh viên', 0.1, pc.sequential.RdBu)
                else:
                    st.warning('Không có sinh viên nào có xếp loại tốt nghiệp là ' + xloksa2 + ' trong tất cả các khoa')
    else:
        df_khoa_overall = df_org[df_org['khoa'] == khoa_overall]
        st.write('Mời bạn chọn các chức năng sau:')
        oks1 = st.checkbox('Xem danh sách sinh viên với xếp loại tốt nghiệp', key='oks1')
        oks2 = st.checkbox('Biểu đồ thống kê xếp loại tốt nghiệp của khoa', key='oks2')
        oks3 = st.checkbox('Biểu đồ so sánh xếp loại tốt nghiệp của các lớp trong khoa', key='oks3')
        st.divider()
        if oks1:
            xloks1 = st.selectbox('Chọn loại tốt nghiệp:', ['Tất cả', 'Rớt', 'Trung bình', 'Trung bình khá', 'Khá', 'Giỏi', 'Xuất sắc'], key='xloks1')
            if xloks1 != 'Tất cả':
                df_oks1 = df_khoa_overall[df_khoa_overall['xeploai'] == xloks1]
                csv_oks1 = df_oks1.to_csv(index=False)
            else:
                df_oks1 = df_khoa_overall.copy()
                csv_oks1 = df_oks1.to_csv(index=False)
            st.dataframe(df_oks1[needed_cols35])
            st.download_button(label='Tải xuống danh sách sinh viên', data=csv_oks1, file_name='DS_sinh_vien.csv', mime='text/csv', key='dl_os_one_1')
        if oks2:
            df_oks2 = df_khoa_overall['xeploai'].value_counts().reset_index()
            df_oks2.columns = ['Xếp loại', 'Số lượng']
            make_bar_chart(df_oks2, 'Xếp loại', 'Số lượng', 0, 'Số lượng', 'Kết quả xếp loại tốt nghiệp của khoa ' + khoa_overall, 'Xếp loại', 'Số lượng sinh viên', 0.2, pc.sequential.Brwnyl)
        if oks3:
            xloks3 = st.selectbox('Chọn loại tốt nghiệp:', ['Tất cả', 'Rớt', 'Trung bình', 'Trung bình khá', 'Khá', 'Giỏi', 'Xuất sắc'], key='xloks3')
            if xloks3 == 'Tất cả':
                df_oks3 = df_khoa_overall.groupby(['lopsh', 'xeploai']).size().reset_index(name='count')
                make_bar_chart(df_oks3, 'lopsh', 'count', 'xeploai', 'count', 'Biểu đồ so sánh xếp loại tốt nghiệp của tất cả các lớp trong khoa', 'Lớp sinh hoạt', 'Số lượng sinh viên', 0.1, pc.sequential.RdBu)
            else:
                df_oks3 = df_khoa_overall.groupby(['lopsh', 'xeploai']).size().reset_index(name='count')
                df_oks3 = df_oks3[df_oks3['xeploai'] == xloks3]
                if len(df_oks3) != 0:
                    make_bar_chart(df_oks3, 'lopsh', 'count', 0, 'count', 'Biểu đồ so sánh xếp loại tốt nghiệp là ' + xloks3 + ' của tất cả các lớp trong khoa', 'Lớp sinh hoạt', 'Số lượng sinh viên', 0.1, pc.sequential.RdBu)
                else:
                    st.warning('Không có sinh viên nào có xếp loại tốt nghiệp là ' + xloks3 + ' trong khoa ' + khoa_overall)
        st.divider()
        lop_overall = st.selectbox('Mời bạn chọn 1 lớp sinh hoạt của sinh viên:', [u for u in df_khoa_overall['lopsh'].unique()], key='lop_overall')
        if lop_overall:
            df_lop_overall = df_khoa_overall[df_khoa_overall['lopsh'] == lop_overall]
            st.write('Mời bạn chọn các chức năng sau:')
            oksl1 = st.checkbox('Xem danh sách sinh viên với xếp loại tốt nghiệp', key='oksl1')
            oksl2 = st.checkbox('Biểu đồ thống kê xếp loại tốt nghiệp của lớp ', key='oksl2')
            st.divider()
            if oksl1:
                xloksl1 = st.selectbox('Chọn loại tốt nghiệp:', ['Tất cả', 'Rớt', 'Trung bình', 'Trung bình khá', 'Khá', 'Giỏi', 'Xuất sắc'], key='xloksl1')
                if xloksl1 != 'Tất cả':
                    df_oksl1 = df_lop_overall[df_lop_overall['xeploai'] == xloksl1]
                    csv_oksl1 = df_oksl1.to_csv(index=False)
                else:
                    df_oksl1 = df_lop_overall.copy()
                    csv_oksl1 = df_oksl1.to_csv(index=False)
                st.dataframe(df_oksl1[needed_cols35])
                st.download_button(label='Tải xuống danh sách sinh viên', data=csv_oksl1, file_name='DS_sinh_vien.csv', mime='text/csv', key='dl_os_one_lop_1')
            if oksl2:
                df_oksl2 = df_lop_overall['xeploai'].value_counts().reset_index()
                df_oksl2.columns = ['Xếp loại', 'Số lượng']
                make_bar_chart(df_oksl2, 'Xếp loại', 'Số lượng', 0, 'Số lượng', 'Kết quả xếp loại tốt nghiệp của lớp ' + lop_overall, 'Xếp loại', 'Số lượng sinh viên', 0.2, pc.sequential.Emrld)
        
if target_search_s:
    mssv_all = st.text_input('Mời bạn nhập mã số sinh viên cần tra cứu:', key='mssv_all')
    if mssv_all:
        if mssv_all in df_org['mssv'].unique():
            df_mssv_all = df_org[needed_cols35]
            st.dataframe(df_mssv_all[df_mssv_all['mssv'] == mssv_all])
        else:
            st.warning('Mã số sinh viên bạn nhập không tồn tại')   
    
        
# II. Dự đoán với sinh viên chưa tốt nghiệp

# 1. Dự đoán xếp loại bằng mô hình

df_y1['xeploai'] = predict(df_y1, df_org, model_y1, predict_cols1)
df_y2['xeploai'] = predict(df_y2, df_org, model_y2, predict_cols2)
df_y3['xeploai'] = predict(df_y3, df_org, model_y3, predict_cols3)
df_y35['xeploai'] = predict(df_y35, df_org, model_y35, predict_cols35)

# 2. Các chức năng cụ thể
        
with st.sidebar:
    st.divider()
    predict_all = st.checkbox('Dự đoán với sinh viên chưa tốt nghiệp')

if predict_all:
    with st.sidebar:
        y_opt = st.selectbox('Mời bạn chọn sinh viên thuộc các diện sau:', ['Đang học năm 2 kỳ 1', 'Đang học năm 3 kỳ 1', 'Đang học năm 4 kỳ 1', 'Đang học năm 4 kỳ 2'])
        if y_opt == 'Đang học năm 2 kỳ 1':
            df_main = df_y1.copy()
            needed_col = needed_cols1
            predict_col = predict_cols1
        elif y_opt == 'Đang học năm 3 kỳ 1':
            df_main = df_y2.copy()
            needed_col = needed_cols2
            predict_col = predict_cols2
        elif y_opt == 'Đang học năm 4 kỳ 1':
            df_main = df_y3.copy()
            needed_col = needed_cols3
            predict_col = predict_cols3
        elif y_opt == 'Đang học năm 4 kỳ 2':
            df_main = df_y35.copy()
            needed_col = needed_cols35
            predict_col = predict_cols35
        st.write('Mời bạn chọn phạm vi tra cứu:')
        overall_search = st.checkbox('Tổng quát')
        target_search = st.checkbox('Sinh viên cụ thể')

if overall_search:
    khoa = st.selectbox('Mời bạn chọn khoa:', [u for u in list(np.append(np.array('Toàn khoa'), df_main['khoa'].unique()))])
    if khoa == 'Toàn khoa':
        st.write('Mời bạn chọn các chức năng sau:')
        os_all_1 = st.checkbox('Xem danh sách sinh viên với xếp loại tốt nghiệp dự đoán', key='os_all_1')
        os_all_2 = st.checkbox('Biểu đồ thống kê xếp loại tốt nghiệp dự đoán')
        os_all_3 = st.checkbox('Biểu đồ so sánh điểm trung bình giữa các khoa')
        os_all_4 = st.checkbox('Biểu đồ so sánh xếp loại tốt nghiệp dự đoán giữa các khoa')
        st.divider()
        if os_all_1:
            xl1 = st.selectbox('Chọn loại tốt nghiệp:', ['Tất cả', 'Rớt', 'Trung bình', 'Trung bình khá', 'Khá', 'Giỏi', 'Xuất sắc'], key='xl1')
            if xl1 != 'Tất cả':
                df_show_1 = df_main[df_main['xeploai'] == xl1]
                csv_os_all_1 = df_show_1.to_csv(index=False)
            else:
                df_show_1 = df_main.copy()
                csv_os_all_1 = df_show_1.to_csv(index=False)
            st.dataframe(df_show_1[needed_col])
            st.download_button(label='Tải xuống danh sách sinh viên', data=csv_os_all_1, file_name='DS_sinh_vien.csv', mime='text/csv')
        if os_all_2:
            df_os_all_2 = df_main['xeploai'].value_counts().reset_index()
            df_os_all_2.columns = ['Xếp loại', 'Số lượng']
            make_bar_chart(df_os_all_2, 'Xếp loại', 'Số lượng', 0, 'Số lượng', 'Kết quả xếp loại tốt nghiệp dự đoán của tất cả các khoa', 'Xếp loại', 'Số lượng sinh viên', 0.2, pc.sequential.Brwnyl)
        if os_all_3:
            df_os_all_3 = df_main.groupby('khoa')['dtb_toankhoa'].mean().reset_index()

            df_os_all_3 = df_os_all_3.sort_values(by='dtb_toankhoa', ascending=False)
            df_os_all_3['dtb_toankhoa'] = df_os_all_3['dtb_toankhoa'].round(2)

            make_bar_chart(df_os_all_3, 'khoa', 'dtb_toankhoa', 0, 'dtb_toankhoa', 'Kết quả học tập của mỗi khoa', 'Khoa', 'Điểm trung bình', 0.3, pc.sequential.Agsunset)
        if os_all_4:
            xl2 = st.selectbox('Chọn loại tốt nghiệp:', ['Tất cả', 'Rớt', 'Trung bình', 'Trung bình khá', 'Khá', 'Giỏi', 'Xuất sắc'], key='xl2')
            if xl2 == 'Tất cả':
                df_os_all_4 = df_main.groupby(['khoa', 'xeploai']).size().reset_index(name='count')
                make_bar_chart(df_os_all_4, 'khoa', 'count', 'xeploai', 'count', 'Biểu đồ so sánh xếp loại tốt nghiệp dự đoán của tất cả các khoa', 'Khoa', 'Số lượng sinh viên', 0.1, pc.sequential.RdBu)
            else:
                df_os_all_4 = df_main.groupby(['khoa', 'xeploai']).size().reset_index(name='count')
                df_os_all_4 = df_os_all_4[df_os_all_4['xeploai'] == xl2]
                if len(df_os_all_4) != 0:
                    make_bar_chart(df_os_all_4, 'khoa', 'count', 0, 'count', 'Biểu đồ so sánh xếp loại tốt nghiệp dự đoán là ' + xl2 + ' của tất cả các khoa', 'Khoa', 'Số lượng sinh viên', 0.1, pc.sequential.RdBu)
                else:
                    st.warning('Không có sinh viên nào có xếp loại tốt nghiệp dự đoán là ' + xl2 + ' trong tất cả các khoa')
    else:
        df_khoa = df_main[df_main['khoa'] == khoa]
        st.write('Mời bạn chọn các chức năng sau:')
        os_one_1 = st.checkbox('Xem danh sách sinh viên với xếp loại tốt nghiệp dự đoán', key='os_one_1')
        os_one_2 = st.checkbox('Biểu đồ thống kê xếp loại tốt nghiệp dự đoán của khoa')
        os_one_3 = st.checkbox('Biểu đồ so sánh xếp loại tốt nghiệp dự đoán của các lớp trong khoa')
        st.divider()
        if os_one_1:
            xl3 = st.selectbox('Chọn loại tốt nghiệp:', ['Tất cả', 'Rớt', 'Trung bình', 'Trung bình khá', 'Khá', 'Giỏi', 'Xuất sắc'], key='xl3')
            if xl3 != 'Tất cả':
                df_show_2 = df_khoa[df_khoa['xeploai'] == xl3]
                csv_os_one_1 = df_show_2.to_csv(index=False)
            else:
                df_show_2 = df_khoa.copy()
                csv_os_one_1 = df_show_2.to_csv(index=False)
            st.dataframe(df_show_2[needed_col])
            st.download_button(label='Tải xuống danh sách sinh viên', data=csv_os_one_1, file_name='DS_sinh_vien.csv', mime='text/csv', key='dl_os_one_1')
        if os_one_2:
            df_os_one_2 = df_khoa['xeploai'].value_counts().reset_index()
            df_os_one_2.columns = ['Xếp loại', 'Số lượng']
            make_bar_chart(df_os_one_2, 'Xếp loại', 'Số lượng', 0, 'Số lượng', 'Kết quả xếp loại tốt nghiệp dự đoán của khoa ' + khoa, 'Xếp loại', 'Số lượng sinh viên', 0.2, pc.sequential.Brwnyl)
        if os_one_3:
            xl4 = st.selectbox('Chọn loại tốt nghiệp:', ['Tất cả', 'Rớt', 'Trung bình', 'Trung bình khá', 'Khá', 'Giỏi', 'Xuất sắc'], key='xl4')
            if xl4 == 'Tất cả':
                df_os_one_3 = df_khoa.groupby(['lopsh', 'xeploai']).size().reset_index(name='count')
                make_bar_chart(df_os_one_3, 'lopsh', 'count', 'xeploai', 'count', 'Biểu đồ so sánh xếp loại tốt nghiệp dự đoán của tất cả các lớp trong khoa', 'Lớp sinh hoạt', 'Số lượng sinh viên', 0.1, pc.sequential.RdBu)
            else:
                df_os_one_3 = df_khoa.groupby(['lopsh', 'xeploai']).size().reset_index(name='count')
                df_os_one_3 = df_os_one_3[df_os_one_3['xeploai'] == xl4]
                if len(df_os_one_3) != 0:
                    make_bar_chart(df_os_one_3, 'lopsh', 'count', 0, 'count', 'Biểu đồ so sánh xếp loại tốt nghiệp dự đoán là ' + xl4 + ' của tất cả các lớp trong khoa', 'Lớp sinh hoạt', 'Số lượng sinh viên', 0.1, pc.sequential.RdBu)
                else:
                    st.warning('Không có sinh viên nào có xếp loại tốt nghiệp dự đoán là ' + xl4 + ' trong khoa ' + khoa)
        st.divider()
        lop = st.selectbox('Mời bạn chọn 1 lớp sinh hoạt của sinh viên:', [u for u in df_khoa['lopsh'].unique()])
        if lop:
            df_lop = df_khoa[df_khoa['lopsh'] == lop]
            st.write('Mời bạn chọn các chức năng sau:')
            os_one_lop_1 = st.checkbox('Xem danh sách sinh viên với xếp loại tốt nghiệp dự đoán', key='os_one_lop_1')
            os_one_lop_2 = st.checkbox('Biểu đồ thống kê xếp loại tốt nghiệp dự đoán của lớp ')
            st.divider()
            if os_one_lop_1:
                xl5 = st.selectbox('Chọn loại tốt nghiệp:', ['Tất cả', 'Rớt', 'Trung bình', 'Trung bình khá', 'Khá', 'Giỏi', 'Xuất sắc'], key='xl5')
                if xl5 != 'Tất cả':
                    df_show_3 = df_lop[df_lop['xeploai'] == xl5]
                    csv_os_one_lop_1 = df_show_3.to_csv(index=False)
                else:
                    df_show_3 = df_lop.copy()
                    csv_os_one_lop_1 = df_show_3.to_csv(index=False)
                st.dataframe(df_show_3[needed_col])
                st.download_button(label='Tải xuống danh sách sinh viên', data=csv_os_one_lop_1, file_name='DS_sinh_vien.csv', mime='text/csv', key='dl_os_one_lop_1')
            if os_one_lop_2:
                df_os_one_lop_2 = df_lop['xeploai'].value_counts().reset_index()
                df_os_one_lop_2.columns = ['Xếp loại', 'Số lượng']
                make_bar_chart(df_os_one_lop_2, 'Xếp loại', 'Số lượng', 0, 'Số lượng', 'Kết quả xếp loại tốt nghiệp của lớp ' + lop, 'Xếp loại', 'Số lượng sinh viên', 0.2, pc.sequential.Emrld)

                

if target_search:
    mssv = st.text_input('Mời bạn nhập mã số sinh viên cần tra cứu:', key='mssv')
    if mssv:
        if mssv in df_main['mssv'].unique():
            df_mssv = df_main[needed_col]
            st.dataframe(df_mssv[df_mssv['mssv'] == mssv])
        else:
            st.warning('Mã số sinh viên bạn nhập không tồn tại')    
