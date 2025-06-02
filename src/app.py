import os
import ast
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import base64

from shingling import Shingles
from get_similarity import Get_Similarity
from task_frequency import load_data, recommend_keyword
from task_poetry import load_model, recommend_poetry
from task_word import load_graph, recommend_next_char
from task_meaning import build_vectorstore, recommend_sentences



def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# ---------------------- 数据加载函数 ----------------------
@st.cache_resource
def load_all_data(style):
    df = load_data(style.lower())
    p, s = load_model(style.lower())

    embed_model = HuggingFaceEmbeddings(
                model_name=f'maidalun1020/bce-embedding-base_v1',
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'batch_size': 4, 'normalize_embeddings': False}
            )
    vec = build_vectorstore(load_flag=True, class_name=style.lower(), embed_model=embed_model)
    
    pr, G = load_graph(style.lower())  
    
    return df, p, s, vec, pr, G

if __name__ == "__main__":
    # ---------------------- Streamlit 界面 ----------------------
    st.set_page_config(page_title="六砚·字库", layout="wide")

    # ---------------------- 嵌入网页 ----------------------
    # with open("./index.html", "r", encoding="utf-8") as f:
    #     html_content = f.read()

    # st.markdown("### 🌐 展示页面")
    # components.html(html_content, height=600, scrolling=True)

    img_path = "./background3.png"  # ./background1.jpg
    color = "#0d1117" # 科技感蓝黑 "#0d1117" 古风米黄色"#e0d8c3" 
    img_base64 = get_base64_of_bin_file(img_path)
    
    background_css = f"""
        <style>
        /* 设置背景图和整体页面样式 */
        .stApp {{
            background-image: url("data:image/jpg;base64,{img_base64}") ;
            background-size: cover;
            background-attachment: fixed;
            background-position: center;
            font-family: "KaiTi", "STKaiti", "SimSun", serif;
            color: #2f2f2f;
        }}
        
        /* 底座容器 */
        .base-container {{
            margin: 30px auto;  /* 居中 */
            max-width: 1200px;
            min-height: 100vh;
            background: rgba(10, 25, 40, 0.85); /* 深蓝半透明 */
            border-radius: 20px;
            box-shadow:
                0 0 20px rgba(30, 150, 230, 0.8),
                inset 0 0 30px rgba(30, 150, 230, 0.5);
            padding: 30px 40px;
            backdrop-filter: blur(8px);  /* 背景模糊，增强科技感 */
            border: 1.5px solid rgba(30, 150, 230, 0.6);
            color: #cceeff;
        }}
        
        /* 页面标题 */
        h1, h2, h3 {{
            font-family: "KaiTi", "STKaiti", "SimSun", serif;
            color: #3b3b3b;
        }}

        h2 {{
            font-size: 36px !important;
            text-shadow: 1px 1px 1px #e0d8c3;
        }}

        
        /* 自定义标题 */
        .left-header {{
            font-size: 28px;
            font-weight: bold;
            margin-bottom: 20px;
            color: #4e342e;
        }}

        /* 标签样式 */
        .left-label, label {{
            font-size: 20px !important;
            color: #3e2723 !important;
        }}

        /* 文本输入框 */
        .stTextInput>div>div>input {{
            font-size: 18px !important;
            height: 38px !important;
            border: 1px solid #ccc;
            background-color: rgba(255, 255, 255, 0.9);
        }}


        /* 文本输入区域 */
        .stTextArea > div > textarea {{
            font-size: 18px !important;
            background-color: rgba(255, 255, 255, 0.9);  /* 半透明白色背景 */
            border: 1px solid #ccc !important;           /* 边框颜色 */
            border-radius: 8px;                          /* 圆角边框 */
            padding: 10px;                               /* 内边距，提升舒适度 */
            line-height: 1.5;                            /* 行间距 */
            color: #2f2f2f;                              /* 字体颜色 */
            resize: vertical;                            /* 允许垂直拉伸 */
            box-shadow: 2px 2px 6px rgba(0, 0, 0, 0.1);   /* 轻微阴影效果 */
            transition: border-color 0.3s, box-shadow 0.3s;
            font-family: "KaiTi", "STKaiti", "SimSun", serif;  /* 古风字体 */
        }}

        /* 按钮样式 */
        .stButton>button {{
            font-size: 18px !important;          /* 字体大小设为18像素，`!important`表示强制覆盖其他冲突样式 */
            height: 20px !important;              /* 按钮高度设为40像素，带`!important`确保生效 */
            padding: 10px 20px !important;         /* 内边距，6像素上下，20像素左右，带`!important` */
            background-color: #c5b796 !important;/* 背景色，浅米黄色，带`!important`确保覆盖默认样式 */
            color: black;                         /* 字体颜色设为黑色 */
            border-radius: 5px;                   /* 按钮圆角半径为8像素，圆润效果 */
            border: 1px solid #aaa;               /* 边框为1像素实线，颜色是浅灰色 */
            transition: background-color 0.3s;   /* 背景色变化时，动画过渡时间0.3秒，平滑过渡 */
        }}

        .stButton>button:hover {{
            background-color: #b4a078 !important;
            color: white;
        }}
        
        
        .expander-box {{
            background-color: rgba(245, 250, 255, 0.9); 
            padding: 16px;
            border-radius: 10px;
            border: 1px solid #ccc;
            font-size: 18px;
            font-family: "KaiTi", "STKaiti", "SimSun", serif;
            color: #2f2f2f;
        }}

        /* 正文段落 */
        p {{
            font-size: 20px !important; /* 字体大小设为10像素，`!important`表示强制覆盖其他冲突样式 */
            background-color: rgba(255, 248, 235, 0.0); /* 更古风的米黄色 */
            padding: 10px 25px;  /* 上下10像素，左右20像素内边距 */
            border-radius: 10px; 
            border: 0px solid #e1d3b8;
            box-shadow: 2px 2px 6px rgba(0, 0, 0, 0.00);
            font-family: "KaiTi", "STKaiti", "SimSun", serif;
            color: {color}; /* 字体颜色 */
            line-height: 1.5;
            margin-bottom: 16px;
        }}
        </style>
    """

    
    st.markdown(background_css, unsafe_allow_html=True)
    
    st.markdown("""
        <h2 style='text-align: center;'>古代诗歌辅助创作系统</h2>
        <p style='text-align: center; color: #0d1117; '>由 六砚斋·数据挖掘坊 倾情打造，支持 楚辞 / 唐诗 / 宋词 / 元曲 创作 </p>
    """, unsafe_allow_html=True)

    st.markdown("""
        <style>
        /* 隐藏原始 radio 的圆圈 */
        [data-baseweb="radio"] input[type="radio"] {
            display: none;
        }

        /* 每个选项的容器变成按钮风格 */
        [data-baseweb="radio"] label {
            display: inline-block;
            background-color: #f5f0e6;
            color: #2f2f2f;
            border: 1px solid #c5b796;
            border-radius: 8px;
            padding: 10px 20px;
            margin: 5px;
            cursor: pointer;
            font-size: 18px;
            font-family: "KaiTi", "STKaiti", "SimSun", serif;
            transition: all 0.3s ease-in-out;
        }

        /* 鼠标悬停高亮 */
        [data-baseweb="radio"] label:hover {
            background-color: #e5d4b1;
            color: black;
        }

        /* 当前选中项高亮显示 */
        [data-baseweb="radio"] input[type="radio"]:checked + div {
            background-color: #c5b796;
            color: white;
            border: 2px solid #856404;
            font-weight: bold;
        }
        </style>
        """, unsafe_allow_html=True)
    
    st.markdown("""
        <style>
        .left-header {
            font-size: 32px;
            font-weight: 700;
            color: #8b5e3c; /* 古风茶褐色 */
            font-family: "KaiTi", "STKaiti", "SimSun", serif;
            padding-bottom: 10px;
            border-bottom: 2px solid #c5b796; /* 底部装饰线 */
            margin-bottom: 20px;
            text-shadow: 0.5px 0.5px 0.5px #d7c9a7; /* 柔和阴影 */
        }
        
        input[type="number"] {
            height: 40px;
            font-size: 16px;
        }
        </style>
        """, unsafe_allow_html=True)

    left_col, right_col = st.columns([0.7, 2.3])  # 左栏窄，右栏宽

    with left_col:
        st.markdown("<div class='left-header'>设置</div>", unsafe_allow_html=True)
        # 显示中文体裁选项
        style_zh = ["楚辞", "唐诗", "宋词", "元曲"]
        style_en = ["chuci", "shi", "songci", "yuanqu"]

        style_display = st.selectbox("请选择体裁", style_zh, index=0)
        style = style_en[style_zh.index(style_display)]
        num_recommend = st.number_input("推荐数量", min_value=1, max_value=20, value=5, step=1)
        method = st.radio(
            "选择想使用的功能",
            ["推荐下一个字", "推荐主题词语", "推荐相关诗句", "推荐相关诗篇"],
            index=0
        )

        
    with right_col:
        keyword = st.text_area("开始我的创作", value="", placeholder="如：山川异域，风月同天 / 只因你太美")
        run = st.button("推荐一下")
        

        if run:
            df, p, s, vec, pr, G  = load_all_data(style)
            
            if method == "推荐下一个字":
                result = recommend_next_char(keyword, pr, G, top_k=num_recommend+1)
                # st.success("推荐结果：")
                st.text_area("推荐下一个字", result, height=200)
            
            elif method == "推荐主题词语":
                result = recommend_keyword(keyword, df)
                # st.success("推荐结果：")
                st.text_area("推荐词语", result, height=200)
                
                
            elif method == "推荐相关诗句":
                result = recommend_sentences(keyword, vec, top_n=num_recommend+1)
                # st.success("推荐结果：")
                st.text_area("推荐诗句", result, height=200)
                
            elif method == "推荐相关诗篇":
                result =  recommend_poetry(keyword, p, s, num=num_recommend)
                # st.success("推荐结果：")
                st.text_area("推荐诗篇", result, height=200)
                
                # st.info("该方法暂未实现，请自行补充函数。")
                
    
    # ---------------------- 使用说明 ----------------------
    with st.expander("使用说明", expanded=False):
        st.markdown("""
        <div class="expander-box">
        Hi~ o(*￣▽￣*)ブ 这里是使用说明
        
        **使用方法:**
        - 输入一个关键词（如“山”、“月”、“风”等），系统会分析该体裁中与之经常共现的词语。  
        - 共现关系基于多种方式挖掘（语义or词频相似度）。  
        - 推荐词语适合用于辅助创作、构思诗意意象、模仿风格。

        **体裁说明**  
        - `chuci`：楚辞（屈原创作风格，想象丰富）  
        - `shi`：古诗（以唐诗为主，意境严谨）  
        - `songci`：宋词（婉约或豪放，多描情写景）  
        - `yuanqu`：元曲（戏剧性强，语言通俗）
        </div>
        """, unsafe_allow_html=True)
