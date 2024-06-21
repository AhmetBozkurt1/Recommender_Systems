#############################
# Content Based Recommendation (İçerik Temelli Tavsiye)
#############################

#############################
# İŞ PROBLEMİ: Filmlere ait bir veri setinde kullanıcı izlediği film sonrasında bir film önermek istiyorum
#############################

#############################
# Film Overview'larına Göre Tavsiye Geliştirme
#############################

# 1. TF-IDF Matrisinin Oluşturulması
# 2. Cosine Similarity Matrisinin Oluşturulması
# 3. Benzerliklere Göre Önerilerin Yapılması
# 4. Çalışma Scriptinin Hazırlanması


# Gerekli kütüphaneleri ve ayarlamaları yapıyorum
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings("ignore")

pd.set_option("display.max_columns", None)
pd.set_option("display.max_row", None)
pd.set_option("display.width", 500)
pd.set_option("display.float_format", lambda x: "%.4f" % x)

# VERİ SETİNİN OKUTULMASI
df = pd.read_csv("movies_metadata.csv")
df.head()
df.isnull().sum()
df.shape

# Veri setinde gerekli olan değişkenler aslında film açıklamalarının olduğu "overview" ve film isimlerinin olduğu
# "tile" değişkenleri sadece bunları almam yeterli.
df = df[["title", "overview"]]
df.head()
df.isnull().sum()
df.shape
df.loc[df["title"].isnull()]

# En dataframe'de boş değerler var title boş olanları direkt sileceğim çünkü film isim önermesi yapacağız olmayan ismi
# nasıl önereceğiz diğer tarafta da yorumları boş olanların içini de boş ifade ile dolduracağım.
df = df.dropna(subset=["title"])
df["overview"] = df["overview"].fillna("")

#################################
# 1. TF-IDF Matrisinin Oluşturulması
#################################
tf_idf = TfidfVectorizer(stop_words="english")
tf_idf_matrix = tf_idf.fit_transform(df["overview"])
tf_idf_matrix.shape

# Metinlerin sütünlara gelmesi ile oluşan değişkenleri görmek için matrise get_feature_names_out() methotunu kullanarak ulaşabiliriz
tf_idf.get_feature_names_out()

# Dokuman ve terimlerin kesişimlerinde ne var görmek istersek de toarray() kullanıyoruz
tf_idf_matrix.toarray()

#################################
# 2. COSINE SMILARITY MATRİSİNİN OLUŞTURULMASI
#################################

# Yukarıda metin vektörlerini oluşturduk şimdi matematiksel olarak uzaklık temelli veya benzerlik temelli yaklaşımlardan
# birini kullanarak hangi filmlerin birbirine daha çok benzer olduğunu bulmamız lazım.Burada kosinüs benzerliği formülünü
# kullanacağız benzerlik için ama uzaklık olarak bakacaksak öklit de uygulanabilir.
cosine_sim = cosine_similarity(tf_idf_matrix)
cosine_sim[2]

#################################
# 3. BENZERLİKLERİNE GÖRE ÜRÜN ÖNERİLERİNİN YAPILMASI
#################################
# İlk olarak ürün isimleri ve index numaralarından pandas.Series oluşturalım.
indices = pd.Series(df.index, index=df["title"])
indices.head()

# Oluşturduğumuz pandas Series'te aynı filmden birden fazla var yani bir çoklama var.O yüzden hepsini teke düşürüyorum
indices.index.value_counts().head()
indices = indices.loc[~indices.index.duplicated(keep="last")]

# Bir tane film seçelim.
movie_index = indices["Russell Madness"]

# Seçilen filme ait skorlarla bir dataframe oluşturuyorum.
similarity_score = pd.DataFrame(cosine_sim[movie_index], columns=["score"])
similarity_score.head()

# Bu dataframe skorlara göre büyükten küçüğe sıralayıp ilk 10 skorun index bilgilerini alacağım.
rec_movie = similarity_score.sort_values("score", ascending=False)[1:11].index

# Aldığım bu index numaraları ile film title erişeceğim
df["title"].iloc[rec_movie]

# Tüm bu süreci fonksiyonlaştıracağım.
def calculate_cosine_sim(dataframe, column):
    """
    # TF-IDF vektörleşmesini çalıştırmadan boş değerleri dolduralım
    """
    tf_idf = TfidfVectorizer(stop_words="english")
    tf_idf_matrix = tf_idf.fit_transform(dataframe[column])
    cosine_sim = cosine_similarity(tf_idf_matrix)
    return cosine_sim


def content_recommender(dataframe, movie_name, movie_column, cosine_sim, resh_count=11):
    indices = pd.Series(dataframe.index, index=dataframe[movie_column])
    indices = indices.loc[~indices.index.duplicated(keep="last")]
    movie_index = indices[movie_name]
    similarity_score = pd.DataFrame(cosine_sim[movie_index], columns=["score"])
    rec_movie = similarity_score.sort_values("score", ascending=False)[1:resh_count].index
    end_rec = dataframe[movie_column].iloc[rec_movie]
    return end_rec

cosine_sim = calculate_cosine_sim(df, "overview")

content_recommender(df, "Father of the Bride Part II", "title", cosine_sim)

