###########################################
# Item-Based Collaborative Filtering
###########################################

# Veri seti: https://grouplens.org/datasets/movielens/

###########################################
# İş Problemi
###########################################
# online bir film izleme platformu iş birlikçi filtreleme yöntemleri ile kullanıcılarına beğendikleri filmlere karşılık
# önerilerde bulunacak.Önerilerini kullanıcının beğendiği film ile benzer beğenilme örüntüsüne sahip olan diğer filmler
# ile yapmak istiyor

# iki adet veri seti var "movie.csv" veri setinde movieId ve title değişkenleri
# "rating.csv" veri setinde userid,movieId,rating,timestamp. bu iki veri setinde "movieId" ortaktır. biz rating.csv
# veri setinde benzerlikleri hesaplayacağız movie.csv veri setinde sadece filmlerin isimlerini öğrenmek için kullanabiliriz.
# Ya da iki veri setini birleştiririz.

# Adım 1: Veri Setinin Hazırlanması
# Adım 2: User Movie Df'inin Oluşturulması
# Adım 3: Item-Based Film Önerilerinin Yapılması
# Adım 4: Çalışma Scriptinin Hazırlanması

######################################
# Adım 1: Veri Setinin Hazırlanması
######################################
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)
pd.set_option("display.max_row", None)
pd.set_option("display.width", 500)

# İki ayrı veri setini birleştireceğiz.
movie_df = pd.read_csv("movie.csv")
rating_df = pd.read_csv("rating.csv")
movie_df.head()
rating_df.head()
movie_df.isnull().sum()
rating_df.isnull().sum()

df = pd.merge(movie_df, rating_df, how="left", on="movieId")
df.head()
df.shape
df["movieId"].nunique()

# Yorum sayısı az olan kullanıcıları ve az değerlendirme alan filmleri filtreliyorum.

# UserId için İşlemler 
df["userId"].value_counts().head()
user_filter = pd.DataFrame(df["userId"].value_counts())
user_index = user_filter.loc[user_filter["count"] <= 100].index
new_df = df.loc[~df["userId"].isin(user_index)]

# MovieId için İşlemler
df["movieId"].value_counts().head()
movie_filter = pd.DataFrame(df["movieId"].value_counts())
movie_index = movie_filter.loc[movie_filter["count"] <= 5000].index
new_df = df.loc[~df["movieId"].isin(movie_index)]

# Veri setinde değişkenleri film olarak sütunlarda kullanıcılar olacak şekilde düzenliyorum.
user_movie_df = new_df.pivot_table(index="userId", columns="title", values="rating")
user_movie_df.shape
user_movie_df.iloc[:10,:5]
######################################
# Adım 3: Item-Based Film Önerilerinin Yapılması
#####################################

# İlk olarak bir film seçip ona verilen puanları alıyorum.
movie_name = "12 Angry Men (1957)"
movie_name = user_movie_df[movie_name]

# Burada seçilen film ile korelasyon bakımından yüksek olan filmleri getireceğim.
user_movie_df.corrwith(movie_name).sort_values(ascending=False)[1:10]

# Örneklem çekerek yapalım.
movie_name = user_movie_df.sample(1, axis=1).head().columns.values[0]
movie_name = user_movie_df[movie_name]
user_movie_df.corrwith(movie_name).sort_values(ascending=False)[1:10]
# Bu şekilde filmler arasında benzer beğenilme örüntüsüne sahip filmlere korelasyon yardımıyla ulaşıyorum.

# Film ismi hatırlamadık.Girilen kelimelerle benzer film isimlerini getiren fonksiyon yazalım.
def movie_search(keyword, dataframe):
    keyword = keyword.lower()
    return [col for col in dataframe.columns if keyword in col.lower()]

# Tüm bu süreci fonksiyonlaştıralım.
def item_based_recommnder(dataframe, movie_name, resh_count=10):
    movie_name = dataframe[movie_name]
    rec_movie = dataframe.corrwith(movie_name).sort_values(ascending=False)[1:resh_count].index
    rec_movie = pd.Series(rec_movie)
    return rec_movie

item_based_recommnder(user_movie_df, "12 Angry Men (1957)")










