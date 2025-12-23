import streamlit as st
import pandas as pd
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
import os
import joblib

# Configuration for Matplotlib to display Chinese characters
plt.rcParams['font.sans-serif'] = ['SimHei']  # Use SimHei for Chinese characters
plt.rcParams['axes.unicode_minus'] = False    # Fix minus sign display

import requests
import speech_recognition as sr
import whisper

# from streamlit_audiorecorder import audiorecorder # Removed external dependency
from utils import extract_features, get_gender_from_pitch, load_data
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# Page Config
st.set_page_config(
    page_title="语音情感与人工智能分析系统 (专业版)",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for "Beautiful" UI
def apply_custom_css():
    st.markdown("""
    <style>
        /* Global Settings */
        .main {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        
        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background-image: linear-gradient(to bottom, #EBF5FC, #D6EAF8);
            color: #2c3e50;
            border-right: 1px solid #AED6F1;
        }
        [data-testid="stSidebar"] .css-17lntkn { 
            color: #2c3e50;
        }
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
            color: #2c3e50 !important;
        }
        [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] .stRadio label {
             color: #2c3e50 !important;
        }
        /* Radio Button Base Style */
        [data-testid="stSidebar"] .stRadio label {
            padding: 10px;
            border-radius: 8px;
            margin-bottom: 5px;
            transition: background-color 0.3s;
        }

        /* Hover Effect */
        [data-testid="stSidebar"] .stRadio label:hover {
            background-color: rgba(255, 255, 255, 0.5);
        }

        /* Selected Label Container - Light Blue Background & Left Border */
        [data-testid="stSidebar"] .stRadio label[data-baseweb="radio"][aria-checked="true"] {
            background-color: rgba(33, 150, 243, 0.15) !important;
            border-left: 6px solid #1976D2 !important;
            padding-left: 14px !important; /* Compensate for border width */
        }

        /* Selected Radio Button Circle - Blue Fill */
        [data-testid="stSidebar"] .stRadio label[data-baseweb="radio"][aria-checked="true"] > div:first-child {
            background-color: #1976D2 !important;
            border-color: #1976D2 !important;
            box-shadow: 0 0 5px rgba(25, 118, 210, 0.5);
        }
        
        /* Selected Text - Blue & Bold */
        [data-testid="stSidebar"] .stRadio label[data-baseweb="radio"][aria-checked="true"] p {
            color: #0D47A1 !important;
            font-weight: 800 !important;
            font-size: 1.1em !important;
        }
        
        /* Titles and Headers */
        h1, h2, h3 {
            color: #2c3e50;
            font-weight: 600;
        }
        h1 {
            text-align: center;
            padding-bottom: 20px;
            border-bottom: 2px solid #3498db;
            margin-bottom: 30px;
        }
        
        /* Buttons */
        .stButton>button {
            width: 100%;
            border-radius: 20px;
            height: 3em;
            background: linear-gradient(to right, #4facfe 0%, #00f2fe 100%);
            color: white;
            border: none;
            font-weight: bold;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 8px rgba(0,0,0,0.15);
        }
        
        /* Cards/Containers */
        div.css-1r6slb0 {
            background-color: white;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        }
        
        /* Metrics */
        [data-testid="stMetricValue"] {
            font-size: 2rem;
            color: #e74c3c;
        }
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: white;
            border-radius: 4px 4px 0px 0px;
            gap: 1px;
            padding-top: 10px;
            padding-bottom: 10px;
        }
        .stTabs [aria-selected="true"] {
            background-color: #E3F2FD;
            color: #1565C0;
            border-bottom: 2px solid #1976D2;
        }
    </style>
    """, unsafe_allow_html=True)

apply_custom_css()

# Constants
DATA_DIR = r"d:\语言信号处理\keshe2\archive"
MODEL_PATH = "emotion_model.pkl"
DEEPSEEK_API_KEY = "sk-31baa9eb5d5f4ec78f80d37021f0330c"

# Load Model
@st.cache_resource
def load_emotion_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

model = load_emotion_model()

@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

whisper_model = load_whisper_model()

# Translation Mapping
EMOTION_MAP = {
    "euphoric": "愉悦",
    "joyfully": "快乐",
    "sad": "悲伤",
    "surprised": "惊讶"
}

# Helper Functions
def transcribe_audio(audio_path, language_option="自动检测 (Auto)"):
    # Map option to code
    lang_code = None
    google_lang = 'zh-CN' # Default fallback
    
    if "Chinese" in language_option:
        lang_code = "zh"
        google_lang = 'zh-CN'
    elif "English" in language_option:
        lang_code = "en"
        google_lang = 'en-US'
        
    try:
        # Use Whisper for better accuracy
        # Load audio with librosa to ensure 16kHz and compatibility
        y, _ = librosa.load(audio_path, sr=16000)
        
        # Transcribe
        if lang_code:
            result = whisper_model.transcribe(y, language=lang_code)
        else:
            result = whisper_model.transcribe(y) # Auto detect
            
        text = result["text"]
        
        # If empty, try Google as fallback
        if not text.strip():
            raise Exception("Whisper returned empty text")
            
        return text
    except Exception as e:
        # Fallback to Google Speech Recognition
        # print(f"Whisper failed: {e}, using Google fallback...")
        r = sr.Recognizer()
        try:
            with sr.AudioFile(audio_path) as source:
                audio_data = r.record(source)
                text = r.recognize_google(audio_data, language=google_lang)
                return text
        except sr.UnknownValueError:
            return "无法识别语音内容"
        except sr.RequestError:
            return "API请求失败"
        except Exception as e2:
            return f"Error: {e} | Fallback Error: {e2}"

def call_deepseek(text, emotion_prediction, gender):
    if not text or text.startswith("Error") or text == "无法识别语音内容":
        return "无法进行AI分析，因为语音识别失败。"
    
    url = "https://api.deepseek.com/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
    }
    
    prompt = f"""
    你是一个专业的语音情感分析师和心理咨询师。
    
    用户语音内容："{text}"
    系统预测情感：{emotion_prediction}
    系统预测性别：{gender}
    
    请结合语音内容、预测的情感和性别，进行深度的多模态分析：
    1. 分析说话人的当前情绪状态。
    2. 推测潜在的心理特征或压力来源。
    3. 给出针对性的沟通建议或心理调节建议。
    """
    
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "你是一个专业的语音情感分析师和心理咨询师。"},
            {"role": "user", "content": prompt}
        ],
        "stream": False
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            return f"DeepSeek API Error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Connection Error: {e}"

def plot_waveform(y, sr):
    fig, ax = plt.subplots(figsize=(10, 3))
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set_title("时域波形图(Waveform)")
    plt.tight_layout()
    return fig

def plot_spectrogram(y, sr):
    fig, ax = plt.subplots(figsize=(10, 3))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    img = librosa.display.specshow(D, y_axis='log', x_axis='time', sr=sr, ax=ax)
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title("频谱图(Spectrogram)")
    plt.tight_layout()
    return fig

def plot_mel_spectrogram(y, sr):
    fig, ax = plt.subplots(figsize=(10, 3))
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    S_dB = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000, ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set_title("梅尔频谱图(Mel-Spectrogram)")
    plt.tight_layout()
    return fig

def plot_chroma(y, sr):
    fig, ax = plt.subplots(figsize=(10, 3))
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    img = librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', ax=ax)
    fig.colorbar(img, ax=ax)
    ax.set_title("色度图(Chromagram)")
    plt.tight_layout()
    return fig

def plot_mfcc(y, sr):
    fig, ax = plt.subplots(figsize=(10, 3))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    img = librosa.display.specshow(mfcc, x_axis='time', ax=ax)
    fig.colorbar(img, ax=ax)
    ax.set_title("MFCC特征热力图")
    plt.tight_layout()
    return fig

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712035.png", width=100)
    st.title("🎙️ 语音分析 Pro")
    st.markdown("---")
    st.info("💡 基于增强特征与多模型融合的专业分析平台")
    
    st.markdown("### 🧭 导航菜单")
    page = st.radio("Go to", ["数据集概览与分布", "语音深度分析与AI诊断"], label_visibility="collapsed")
    
    st.markdown("---")
    st.markdown("### ⚙️ 系统状态")
    st.caption(f"✅ 模型状态: {'已加载' if model else '未加载'}")
    st.caption(f"✅ Whisper: {'已加载' if whisper_model else '未加载'}")
    
    st.markdown("---")

if page == "数据集概览与分布":
    st.title("📊 数据集高级分析")
    
    if os.path.exists(DATA_DIR):
        with st.spinner("正在加载和分析数据集..."):
            df = load_data(DATA_DIR)
            st.write(f"**总文件数:** {len(df)}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### 情感类别分布")
                st.bar_chart(df['label'].value_counts())
            
            with col2:
                st.markdown("#### 数据预览")
                st.dataframe(df.head(10), use_container_width=True)
            
            st.markdown("### 🧠 深度 AI 算法全景分析")
            st.write("集成 PCA/t-SNE 降维、K-Means/DBSCAN 聚类、孤立森林异常检测以及多模型对比分析。")
            
            # Initialize session state for AI analysis
            if 'ai_analysis_data' not in st.session_state:
                st.session_state.ai_analysis_data = None

            if st.button("启动全算法引擎"):
                features_list = []
                labels_list = []
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                total = len(df)
                
                # 1. Feature Extraction
                status_text.text("正在提取音频特征 (MFCC, Chroma, Mel, Contrast)...")
                for i, row in df.iterrows():
                    feat = extract_features(file_path=row['path'])
                    if feat is not None:
                        features_list.append(feat)
                        labels_list.append(row['label'])
                    progress_bar.progress((i + 1) / total)
                
                if features_list:
                    X = np.array(features_list)
                    y_labels = np.array(labels_list)
                    
                    # Standardization
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    
                    # Save to session state
                    st.session_state.ai_analysis_data = {
                        'X_scaled': X_scaled,
                        'y_labels': y_labels
                    }
                    st.success("特征提取完成！AI 引擎已就绪。")
                else:
                    st.error("未能提取到有效特征，请检查数据集。")

            # Render analysis if data is available in session state
            if st.session_state.ai_analysis_data is not None:
                data = st.session_state.ai_analysis_data
                X_scaled = data['X_scaled']
                y_labels = data['y_labels']
                
                # Re-calculate X for visualizations that need original shape if needed, 
                # but we stored scaled X.
                # Dimensionality Reduction needs to be re-run or stored? 
                # It's fast enough to re-run for visualization usually, 
                # but caching PCA/t-SNE results would be better if dataset is large.
                # For now, we re-run them to keep code simple, or we could store them too.
                # Given dataset size (likely small/medium), re-running is okay.

                # --- Tab Layout for Analysis ---
                st.markdown("---")
                tab_dim, tab_cluster, tab_anomaly, tab_models, tab_importance = st.tabs([
                    "🌌 降维可视化", "🧩 聚类分析", "🔍 异常检测", "⚔️ 模型竞技场", "🔑 特征解密"
                ])
                
                # Pre-calculate PCA for use in multiple tabs
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X_scaled)
                pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
                pca_df['Emotion'] = y_labels
                pca_df['Emotion_CN'] = pca_df['Emotion'].map(lambda x: EMOTION_MAP.get(x, x))

                # 1. Dimensionality Reduction (PCA & t-SNE)
                with tab_dim:
                    col1, col2 = st.columns(2)
                    
                    # PCA
                    with col1:
                        st.subheader("PCA 线性降维")
                        fig_pca = px.scatter(pca_df, x='PC1', y='PC2', color='Emotion_CN', 
                                         title='PCA 分布图',
                                         hover_data=['Emotion'],
                                         template='plotly_white',
                                         color_discrete_sequence=px.colors.qualitative.Bold)
                        st.plotly_chart(fig_pca, use_container_width=True)
                        st.info(f"PCA 解释方差: {np.sum(pca.explained_variance_ratio_):.2%}")

                    # t-SNE
                    with col2:
                        st.subheader("t-SNE 非线性流形学习")
                        # t-SNE can be slow, maybe cache it in session_state too if needed.
                        # For now, calculate it.
                        if 'tsne_df' not in st.session_state:
                             n_samples = X_scaled.shape[0]
                             perplexity_val = min(30, n_samples - 1) 
                             tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity_val)
                             X_tsne = tsne.fit_transform(X_scaled)
                             tsne_df = pd.DataFrame(data=X_tsne, columns=['Dim1', 'Dim2'])
                             tsne_df['Emotion_CN'] = [EMOTION_MAP.get(label, label) for label in y_labels]
                             st.session_state.tsne_df = tsne_df
                        else:
                             tsne_df = st.session_state.tsne_df
                        
                        fig_tsne = px.scatter(tsne_df, x='Dim1', y='Dim2', color='Emotion_CN',
                                          title='t-SNE 分布图',
                                          template='plotly_white',
                                          color_discrete_sequence=px.colors.qualitative.Bold)
                        st.plotly_chart(fig_tsne, use_container_width=True)
                        st.caption("t-SNE 能更好地展示数据的局部结构和类别分离度。")

                # 2. Clustering (K-Means & DBSCAN)
                with tab_cluster:
                    st.subheader("无监督聚类分析 (Unsupervised Clustering)")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### K-Means (K均值聚类)")
                        n_clusters = len(np.unique(y_labels))
                        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                        clusters_km = kmeans.fit_predict(X_scaled)
                        
                        cluster_df = pca_df.copy()
                        cluster_df['Cluster'] = clusters_km.astype(str)
                        
                        fig_km = px.scatter(cluster_df, x='PC1', y='PC2', color='Cluster', symbol='Emotion_CN',
                                         title=f'K-Means 结果 (K={n_clusters})',
                                         template='plotly_white')
                        st.plotly_chart(fig_km, use_container_width=True)
                        st.metric("K-Means 轮廓系数", f"{silhouette_score(X_scaled, clusters_km):.3f}")

                    with col2:
                        st.markdown("#### DBSCAN (密度聚类)")
                        # DBSCAN parameters usually need tuning
                        dbscan = DBSCAN(eps=5, min_samples=3)
                        clusters_db = dbscan.fit_predict(X_scaled)
                        
                        cluster_df_db = pca_df.copy()
                        cluster_df_db['Cluster'] = clusters_db.astype(str)
                        
                        fig_db = px.scatter(cluster_df_db, x='PC1', y='PC2', color='Cluster', symbol='Emotion_CN',
                                         title='DBSCAN 结果 (自动发现簇)',
                                         template='plotly_white')
                        st.plotly_chart(fig_db, use_container_width=True)
                        n_noise = list(clusters_db).count(-1)
                        st.metric("DBSCAN 发现的簇数量", f"{len(set(clusters_db)) - (1 if -1 in clusters_db else 0)}")
                        st.caption(f"注: 标签为 -1 的点被视为噪声点 (共 {n_noise} 个)")

                # 3. Anomaly Detection (Isolation Forest)
                with tab_anomaly:
                    st.subheader("异常检测 (Anomaly Detection)")
                    st.write("使用孤立森林 (Isolation Forest) 算法识别数据集中的异常样本或离群点。")
                    
                    contamination = st.slider("预计异常比例 (Contamination)", 0.01, 0.20, 0.05, 0.01)
                    
                    iso_forest = IsolationForest(contamination=contamination, random_state=42)
                    outliers = iso_forest.fit_predict(X_scaled)
                    
                    anomaly_df = pca_df.copy()
                    anomaly_df['Type'] = np.where(outliers == -1, '异常 (Anomaly)', '正常 (Normal)')
                    
                    fig_anom = px.scatter(anomaly_df, x='PC1', y='PC2', color='Type', symbol='Emotion_CN',
                                      title=f'孤立森林异常检测结果 (异常比例: {contamination})',
                                      color_discrete_map={'异常 (Anomaly)': 'red', '正常 (Normal)': 'lightgrey'},
                                      template='plotly_white',
                                      hover_data=['Emotion_CN'])
                    st.plotly_chart(fig_anom, use_container_width=True)
                    
                    if -1 in outliers:
                        st.warning(f"检测到 {list(outliers).count(-1)} 个潜在的异常样本，这些样本可能包含杂音或标记错误。")

                # 4. Model Comparison
                with tab_models:
                    st.subheader("模型竞技场 (Model Comparison)")
                    st.write("对比不同机器学习算法在该数据集上的分类性能 (80% 训练, 20% 测试)。")
                    
                    if st.button("开始模型对决"):
                        with st.spinner("正在训练模型并进行评估..."):
                            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_labels, test_size=0.2, random_state=42)
                            
                            models = {
                                "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
                                "SVM (RBF Kernel)": SVC(kernel='rbf', probability=True),
                                "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
                                "Naive Bayes": GaussianNB()
                            }
                            
                            results = []
                            
                            model_cols = st.columns(len(models))
                            
                            for idx, (name, clf) in enumerate(models.items()):
                                clf.fit(X_train, y_train)
                                y_pred = clf.predict(X_test)
                                acc = accuracy_score(y_test, y_pred)
                                results.append({"Model": name, "Accuracy": acc})
                                with model_cols[idx]:
                                    st.metric(name, f"{acc:.2%}")
                            
                            res_df = pd.DataFrame(results).sort_values(by="Accuracy", ascending=False)
                            fig_res = px.bar(res_df, x="Accuracy", y="Model", orientation='h', 
                                         title="各模型准确率排行榜", color="Accuracy",
                                         color_continuous_scale="Blues", text_auto='.2%')
                            st.plotly_chart(fig_res, use_container_width=True)

                # 5. Feature Importance
                with tab_importance:
                    st.subheader("特征重要性解密")
                    rf = RandomForestClassifier(n_estimators=100, random_state=42)
                    rf.fit(X_scaled, y_labels)
                    
                    importances = rf.feature_importances_
                    feature_groups = {
                        "MFCC (倒谱系数)": importances[0:80].sum(),
                        "Chroma (色度)": importances[80:104].sum(),
                        "Mel Spectrogram (梅尔频谱)": importances[104:360].sum(),
                        "Spectral Contrast (光谱对比度)": importances[360:].sum()
                    }
                    
                    imp_df = pd.DataFrame(list(feature_groups.items()), columns=['Feature Type', 'Importance'])
                    imp_df = imp_df.sort_values(by='Importance', ascending=False)
                    
                    fig_imp = px.bar(imp_df, x='Importance', y='Feature Type', orientation='h',
                                 title='各类语音特征的重要性贡献度',
                                 color='Importance',
                                 color_continuous_scale='Viridis')
                    st.plotly_chart(fig_imp, use_container_width=True)
            
            if st.button("清空分析缓存"):
                del st.session_state.ai_analysis_data
                if 'tsne_df' in st.session_state:
                    del st.session_state.tsne_df
                st.rerun()

elif page == "语音深度分析与AI诊断":
    st.title("🧠 语音深度分析与AI诊断")
    st.markdown("上传音频或实时录音，系统将进行多维度信号处理、情感/性别识别，并利用DeepSeek AI进行深度心理分析。")
    
    input_method = st.radio("选择输入方式", ["上传文件", "实时录音"], horizontal=True)
    
    audio_path = None
    
    if input_method == "上传文件":
        uploaded_file = st.file_uploader("上传音频文件 (WAV, MP3)", type=["wav", "mp3", "mpeg"])
        if uploaded_file:
            with open("temp_upload.wav", "wb") as f:
                f.write(uploaded_file.getbuffer())
            audio_path = "temp_upload.wav"
            st.audio(audio_path)
            
    elif input_method == "实时录音":
        st.write("点击下方按钮开始录音：")
        # Use native st.audio_input (requires Streamlit >= 1.40)
        audio_buffer = st.audio_input("请录音")
        
        if audio_buffer:
            # Save the recorded file
            with open("temp_record.wav", "wb") as f:
                f.write(audio_buffer.getbuffer())
            audio_path = "temp_record.wav"
            st.success("录音完成！")
    
    if audio_path:
        # Language Selection
        st.markdown("### 🛠️ 分析设置")
        language_option = st.selectbox(
            "选择语音语言 (Select Language)",
            ["自动检测 (Auto)", "中文 (Chinese)", "英文 (English)"],
            index=0,
            help="选择音频的主要语言，'自动检测'通常效果最好，但在杂音较多时指定语言更准确。"
        )

        if st.button("开始全维智能分析"):
            with st.spinner("正在进行多维信号处理、特征提取和AI推理..."):
                # Load Audio
                y, sample_rate = librosa.load(audio_path)
                
                # --- Section 1: Traditional Signal Processing Visualization ---
                st.header("1. 多维信号可视化图谱")
                
                tab1, tab2, tab3, tab4, tab5 = st.tabs(["波形图", "语谱图", "梅尔频谱", "色度图", "MFCC热力图"])
                
                with tab1:
                    st.pyplot(plot_waveform(y, sample_rate))
                with tab2:
                    st.pyplot(plot_spectrogram(y, sample_rate))
                with tab3:
                    st.pyplot(plot_mel_spectrogram(y, sample_rate))
                    st.caption("梅尔频谱图更符合人耳听觉特性，展示了不同频率上的能量分布。")
                with tab4:
                    st.pyplot(plot_chroma(y, sample_rate))
                    st.caption("色度图展示了音频中的音高类别（C, C#, D...），有助于分析音调特征。")
                with tab5:
                    st.pyplot(plot_mfcc(y, sample_rate))
                    st.caption("MFCC（梅尔频率倒谱系数）是语音识别中最核心的特征。")
                
                # --- Section 2: Model Prediction ---
                st.header("2. 智能识别结果")
                col_a, col_b = st.columns(2)
                
                # Gender
                gender = get_gender_from_pitch(audio_path)
                col_a.metric("识别性别", gender, delta="基于基频分析")
                
                # Emotion (Model)
                prediction_cn = "未知"
                if model:
                    features = extract_features(audio_path)
                    if features is not None:
                        # Ensure features shape matches model input (model expects 2D array)
                        prediction = model.predict([features])[0]
                        
                        # Translate prediction to Chinese
                        prediction_cn = EMOTION_MAP.get(prediction, prediction)
                        
                        # Get probability if supported
                        if hasattr(model, "predict_proba"):
                            proba = model.predict_proba([features])[0]
                            max_prob = np.max(proba)
                            col_b.metric("识别情感", prediction_cn, delta=f"置信度: {max_prob:.2%}")
                            
                            # Show prob chart
                            classes = model.classes_
                            # Map classes to CN
                            classes_cn = [EMOTION_MAP.get(c, c) for c in classes]
                            prob_df = pd.DataFrame({"情感": classes_cn, "概率": proba})
                            st.bar_chart(prob_df.set_index("情感"))
                        else:
                            col_b.metric("识别情感", prediction_cn)
                    else:
                        col_b.metric("识别情感", "特征提取失败")
                else:
                    col_b.error("模型未加载")
                
                # --- Section 3: AI Analysis (DeepSeek) ---
                st.header("3. DeepSeek AI 深度心理报告")
                
                # Transcribe
                text = transcribe_audio(audio_path, language_option)
                st.info(f"**语音转文字内容:** {text}")
                
                if text and text != "无法识别语音内容":
                    analysis = call_deepseek(text, prediction_cn, gender)
                    st.success("**AI 心理分析专家报告:**")
                    st.markdown(analysis)
                else:
                    st.warning("未能识别出有效语音内容，无法进行AI深度分析。请尝试清晰说话。")

# Cleanup temp files
# (Optional: In a real app, use tempfile module)
