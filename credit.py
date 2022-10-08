from copyreg import pickle
import json
from flask import Flask, request, render_template, jsonify
import pandas as pd 
import numpy as np 
import pickle
import re
app = Flask(__name__)

###
# 產生 / 頁面
@app.route('/')
def index():
    return render_template('index.html')   # 把 interaction.html 作為 '/' 的模板

# 得到參數跑機器學習後的結果
def predict(a1, a2, a3, a4, a5):
    # 開啟 .pickle，得到 XGB 模型訓練後的結果
    with open('pickle/XGB_model.pickle', 'rb') as fr:
        xgb = pickle.load(fr)
        # 將參數設為 (1, 5) 的矩陣才能夠以類似 df 的形式跑模型
        y_test = np.array([a1, a2, a3, a4, a5]).reshape(1, 5)
    # 得到預測結果
    return xgb.predict(y_test)

# 檢查參數是不是 float
def check_float(arg):
    try:
        arg = str(float(arg))
    except ValueError:
        return False 

# 檢查參數是不是 int
def check_int(arg):
    try:
        arg = str(int(arg))
    except ValueError:
        return False

# 產生 /data 頁面
# /data 在處理參數的部分和 /output 完全相同，因為是測試用的頁面，實際上並不會有連結通往這個頁面
@app.route('/data')
def data():
    # 取得在 index 頁面中得到的 arg1 參數
    arg1 = request.args.get('arg1')
    # if arg1 == '':
    #     return '<h1><b>Education Type<b></h1><br><h1><b>Value Error:<b>Please choose an option</h1>'
    
    # arg1 屬於類別變數，若輸入的變數不在 list 內，則返回錯誤(因為可以直接從網址輸入參數)
    if arg1 not in ['Secondary / secondary special', 'Higher education', 'Incomplete higher', 'Lower secondary', 'Academic degree']:
        return '<h1><b>Education Type<b></h1><br><h1><b>Value Error:<b>Please choose an option</h1>'

    arg2 = request.args.get('arg2')
    # arg2 屬於連續數值，需要使用者自行輸入，因此若輸入空字串或不輸入，則返回錯誤
    if arg2 == '' or arg2 == None:
        return '<h1><b>External Source 1<b></h1><h1><b>Value Error:<b>Please enter floating point between 0 and 1</h1>'
    # 確認 arg2 是否為浮點數(先確認是否為浮點數，再限制使用者輸入的區間值)
    if check_float(arg2) == False:
        return '<h1><b>External Source 1<b></h1><h1><b>Value Error:<b>Please enter floating point between 0 and 1</h1>'
    # 限制使用者輸入區間(此時的狀態已經確認 arg2 為浮點數，因此可直接用函數 float(arg2)，否則會 ValueError)
    if float(arg2) > 1 or float(arg2) < 0:
        return '<h1><b>External Source 1<b></h1><h1><b>Value Error:<b>Please enter floating point between 0 and 1</h1>'

    # 同 arg2
    arg3 = request.args.get('arg3')
    if arg3 == '' or arg3 == None:
        return '<h1><b>External Source 2<b></h1><h1><b>Value Error:<b>Please enter floating point between 0 and 1</h1>'
    if check_float(arg3) == False:
        return '<h1><b>External Source 2<b></h1><h1><b>Value Error:<b>Please enter floating point between 0 and 1</h1>'
    if float(arg3) > 1 or float(arg3) < 0:
        return '<h1><b>External Source 2<b></h1><h1><b>Value Error:<b>Please enter floating point between 0 and 1</h1>'

    arg4 = request.args.get('arg4')
    if arg4 == '' or arg4 == None:
        return '<h1><b>Employed Years<b></h1><h1><b>Value Error:<b>Please enter positive integer</h1>'
    # arg4 屬於非連續數值，檢查是否為整數(就算輸入小數也可以)
    if check_int(arg4) == False:
        return '<h1><b>Employed Years<b></h1><h1><b>Value Error:<b>Please enter positive integer</h1>'
    if int(arg4) > 50:
        return '<h1><b>Employed Years<b></h1><h1><b>Value Error:<b>Please enter positive integer between 0 and 50</h1>'

    # 下拉式選單可以設計一個 <option value="">請選擇</option>
    # 同 arg1
    arg5 = request.args.get('arg5')
    if arg5 not in ['1', '2', '3']:
        return '<h1><b>Region Rating<b></h1><h1><b>Value Error:<b>Please choose an option</h1>'

    # 打開 .pickle 檔取得字典，用取得的參數對應轉換後的數值來跑模型
    with open('pickle/edu_arg.pickle', 'rb') as fr2:
        edu_arg = pickle.load(fr2)
    arg1_1 = edu_arg[arg1]

    # 取得最小值和全距，對 arg2,3 進行處理
    with open('pickle/max_min.pickle', 'rb') as fr1:
        max_min = pickle.load(fr1)
    arg2_1 = ((float(arg2)-max_min.iloc[0, 0])/max_min.iloc[1, 0])*0.6+0.2
    arg3_1 = ((float(arg3)-max_min.iloc[0, 1])/max_min.iloc[1, 1])*0.6+0.2

    # 先將輸入的數值進行分組，再用字典進行對應
    with open('pickle/emp_arg.pickle', 'rb') as fr3:
        emp_arg = pickle.load(fr3)
    list_slice = [int(i) for i in np.linspace(0, 50, num=11)]
    list_slice.append(1050)
    for j in list_slice:
        if float(arg4) == 50:
            arg4_1 = emp_arg[pd.Interval(45, 50, closed='right')]
            break
        elif float(arg4) >= j:
            continue
        elif float(arg4) < j:
            arg4_1 = emp_arg[pd.Interval(j-5, j, closed='right')]
            break
        else:
            arg4_1 = emp_arg[pd.Interval(45, 50, closed='right')]
            break

    with open('pickle/reg_arg.pickle', 'rb') as fr4:
        reg_arg = pickle.load(fr4)
    arg5_1 = reg_arg[int(arg5)]

    # 取得機器學習預測的結果
    target = str(predict(arg1_1, arg2_1, arg3_1, arg4_1, arg5_1))[1]
    
    # 用字典的形式顯示在 /data 頁面上
    dict_a1 = {'EDUCATION_TYPE': arg1_1}
    dict_a2 = {'EXT_SOURCE_2': arg2_1}
    dict_a3 = {'EXT_SOURCE_3': arg3_1}
    dict_a4 = {'YEARS_EMP_BINNED': arg4_1}
    dict_a5 = {'REGION_RATING': arg5_1}
    target = str(predict(arg1_1, arg2_1, arg3_1, arg4_1, arg5_1))[1]
    # return f'<h1>{target}</h1>'
    if target == '1':
        dict1 = {'TARGET':1}
        list1 = [dict_a1, dict_a2, dict_a3, dict_a4, dict_a5, dict1]
        return jsonify(list1)
    else:
        dict0 = {'TARGET': 0}
        list0 = [dict_a1, dict_a2, dict_a3, dict_a4, dict_a5, dict0]
        return jsonify(list0)


# 產生 /output 頁面(顯示結果的頁面)
@app.route('/output')
def output():
    arg1 = request.args.get('arg1')
    # if arg1 == '':
    #     return '<h1><b>Education Type<b></h1><br><h1><b>Value Error:<b>Please choose an option</h1>'
    if arg1 not in ['Secondary / secondary special', 'Higher education', 'Incomplete higher', 'Lower secondary', 'Academic degree']:
        return '<h1><b>Education Type<b></h1><br><h1><b>Value Error:<b>Please choose an option</h1>'

    arg2 = request.args.get('arg2')
    if arg2 == '' or arg2 == None:
        return '<h1><b>External Source 1<b></h1><h1><b>Value Error:<b>Please enter floating point between 0 and 1</h1>'
    if check_float(arg2) == False:
        return '<h1><b>External Source 1<b></h1><h1><b>Value Error:<b>Please enter floating point between 0 and 1</h1>'
    if float(arg2) > 1 or float(arg2) < 0:
        return '<h1><b>External Source 1<b></h1><h1><b>Value Error:<b>Please enter floating point between 0 and 1</h1>'

    arg3 = request.args.get('arg3')
    if arg3 == '' or arg3 == None:
        return '<h1><b>External Source 2<b></h1><h1><b>Value Error:<b>Please enter floating point between 0 and 1</h1>'
    if check_float(arg3) == False:
        return '<h1><b>External Source 2<b></h1><h1><b>Value Error:<b>Please enter floating point between 0 and 1</h1>'
    if float(arg3) > 1 or float(arg3) < 0:
        return '<h1><b>External Source 2<b></h1><h1><b>Value Error:<b>Please enter floating point between 0 and 1</h1>'

    arg4 = request.args.get('arg4')
    if arg4 == '' or arg4 == None:
        return '<h1><b>Employed Years<b></h1><h1><b>Value Error:<b>Please enter positive integer</h1>'
    if check_int(arg4) == False:
        return '<h1><b>Employed Years<b></h1><h1><b>Value Error:<b>Please enter positive integer</h1>'
    if int(arg4) > 50:
        return '<h1><b>Employed Years<b></h1><h1><b>Value Error:<b>Please enter positive integer between 0 and 50</h1>'

    arg5 = request.args.get('arg5')   # 下拉式選單可以設計一個 <option value="">請選擇</option>
    if arg5 not in ['1', '2', '3']:
        return '<h1><b>Region Rating<b></h1><h1><b>Value Error:<b>Please choose an option</h1>'

    with open('pickle/edu_arg.pickle', 'rb') as fr2:
        edu_arg = pickle.load(fr2)
    arg1_1 = edu_arg[arg1]

    with open('pickle/max_min.pickle', 'rb') as fr1:
        max_min = pickle.load(fr1)
    arg2_1 = ((float(arg2)-max_min.iloc[0, 0])/max_min.iloc[1, 0])*0.6+0.2
    arg3_1 = ((float(arg3)-max_min.iloc[0, 1])/max_min.iloc[1, 1])*0.6+0.2

    with open('pickle/emp_arg.pickle', 'rb') as fr3:
        emp_arg = pickle.load(fr3)
    list_slice = [int(i) for i in np.linspace(0, 50, num=11)]
    list_slice.append(1050)
    for j in list_slice:
        if float(arg4) == 50:
            arg4_1 = emp_arg[pd.Interval(45, 50, closed='right')]
            break
        elif float(arg4) >= j:
            continue
        elif float(arg4) < j:
            arg4_1 = emp_arg[pd.Interval(j-5, j, closed='right')]
            break
        else:
            arg4_1 = emp_arg[pd.Interval(45, 50, closed='right')]
            break

    with open('pickle/reg_arg.pickle', 'rb') as fr4:
        reg_arg = pickle.load(fr4)
    arg5_1 = reg_arg[int(arg5)]

    target = str(predict(arg1_1, arg2_1, arg3_1, arg4_1, arg5_1))[1]
    
    # 取得 arg2,3 在原本的資料集(已經填補完空值了) 的百分位數
    with open('pickle/alldf.pickle', 'rb') as fr:
        df = pickle.load(fr)
    ext2 = df['EXT_SOURCE_2'].tolist()
    ext2.append(float(arg2))
    ext2 = sorted(ext2)
    ext2_pct = round(ext2.index(float(arg2))/len(ext2)*100)   
    ext3 = df['EXT_SOURCE_3'].tolist()
    ext3.append(float(arg3))
    ext3 = sorted(ext3)
    ext3_pct = round(ext3.index(float(arg3))/len(ext3)*100)
    
    dict_a6 = {'Secondary / secondary special': '國高中', 'Higher education': '學士', 'Incomplete higher': '大學肄業',
               'Lower secondary': '國小', 'Academic degree': '碩士'}
    dict_a7 = {'1': '評級一', '2': '評級二', '3': '評級三'}
    dict_a8 = {'0': '不會違約', '1': '會違約'}

    return render_template('output.html', arg1=dict_a6[arg1], arg2=arg2, arg3=arg3, arg4=arg4, arg5=dict_a7[arg5], target=dict_a8[target],
                           ext2_pct=ext2_pct, ext3_pct=ext3_pct)

# 建立其他頁面對應 .html
@app.route('/choose_model')
def choose_model():
    return render_template('choose_model.html')


@app.route('/Conclusion')
def Conclusion():
    return render_template('Conclusion.html')


@app.route('/data-process')
def data_process():
    return render_template('data-process.html')


@app.route('/data-review')
def data_review():
    return render_template('data-review.html')


@app.route('/interaction')
def interaction():
    return render_template('interaction.html')


@app.route('/machine-learning')
def machine_learning():
    return render_template('machine-learning.html')


# 運行 app
app.run()