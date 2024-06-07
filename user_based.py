############################################
# User-Based Collaborative Filtering
#############################################

###########################################
# İş Problemi
###########################################
# online bir film izleme platformu iş birlikçi filtreleme yöntemleri ile kullanıcılarının izlediklerileri filmlere karşılık
# önerilerde bulunacak.Önerilerini kullanıcının izlediği beğendiği film ile aynı filmi izleyen beğenen, beğenme
# davranışlarında benzerlik gösteren diğer kullanıcıların izledikleri filmi önereceğiz


# Adım 1: Veri Setinin Hazırlanması
# Adım 2: Öneri Yapılacak Kullanıcının İzlediği Filmlerin Belirlenmesi
# Adım 3: Aynı Filmleri İzleyen Diğer Kullanıcıların Verisine ve Id'lerine Erişmek
# Adım 4: Öneri Yapılacak Kullanıcı ile En Benzer Davranışlı Kullanıcıların Belirlenmesi
# Adım 5: Weighted Average Recommendation Score'un Hesaplanması
# Adım 6: Çalışmanın Fonksiyonlaştırılması

#############################################
# Adım 1: Veri Setinin Hazırlanması
#############################################
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)
pd.set_option("display.max_row", None)
pd.set_option("display.width", 500)

# Veri setlerini okutalım.
movie_df = pd.read_csv("/Users/ahmetbozkurt/Desktop/Recommender_Systems/dataset/movie.csv")
rating_df = pd.read_csv("/Users/ahmetbozkurt/Desktop/Recommender_Systems/dataset/rating.csv")
movie_df.head()
rating_df.head()
# İki veri setini de birleştirelim.
df = pd.merge(movie_df, rating_df, how="left", on="movieId")
df.head()

# Veri setinde çok az yorum alan ve çok az yorum yapan kullanıcılar var bunları veri setinden çıkaracağız.
df["title"].value_counts().tail()
comment_counts = pd.DataFrame(df["title"].value_counts())
comment_counts.head()
comment_index = comment_counts.loc[comment_counts["count"] <= 5000].index
comment_movies = df.loc[~df["title"].isin(comment_index)]

df["userId"].value_counts().tail()
comment_users = pd.DataFrame(df["userId"].value_counts())
comment_users.head()
comment_users_index = comment_users.loc[comment_users["count"] <= 100].index
comment_movies = df.loc[~df["userId"].isin(comment_users_index)]
comment_movies.head()

# Yeni bir datafame oluşturuyorum.
user_movie_df = comment_movies.pivot_table(index="userId", columns="title", values="rating")
user_movie_df.iloc[:5, :5]



# Rastgele bir kullanıcı seçeceğim.
random_user = int(user_movie_df.sample(1, axis=0, random_state=21).index[0])

# Belirlenen kullanıcının izlediği filmleri alalım.
random_user_df = user_movie_df.loc[user_movie_df.index == random_user]
random_user_df.iloc[:5, :5]
random_user_df.shape

# İki farklı şekilde buna ulaşırım.
user_watch = random_user_df.dropna(axis=1).columns.to_list()
len(user_watch)

user_watch = random_user_df.columns[random_user_df.notnull().any()].to_list()

#############################################
# Aynı Filmi İzleyen Diğer Kullanıcıların Verisine ve Id'lerine Erişmek
#############################################
# ilk olarak yukarıda kullanıcımın izlediği filmler listesinden bir dataframe oluşturuyorum.
movies_watched_df = user_movie_df[user_watch]
movies_watched_df.shape


# Her kullanıcının izlediği film sayısına ulaşalım. 3 Farklı yöntemle de buna ulaşabiliriz.
movies_watched_df.iloc[:5, :5]
user_movie_count = movies_watched_df.notnull().sum(axis=1)
user_movie_count = user_movie_count.reset_index()
user_movie_count.columns = ["userId", "movie_count"]
user_movie_count.head()

# Diğer yöntemlerle de buna ulaşabiliriz.
user_movie_count = movies_watched_df.T.notnull().sum()
user_movie_count = movies_watched_df.applymap(lambda x: 1 if pd.notnull(x) else 0).sum(axis=1)

# Seçilen user izlediği film sayısından bir değer elde ettim.
perc = len(user_watch) * 60/100

# Daha sonra yakaladığım bu orandan fazlasını izleyen kullanıcılara erişiyorum.
user_same_movies = user_movie_count.loc[user_movie_count["movie_count"] > perc, "userId"]
user_same_movies.head()

#############################################
# Adım 4: Öneri Yapılacak Kullanıcı ile En Benzer Davranışlı Kullanıcıların Belirlenmesi
#############################################

# Yukarıda belirlediğim kullanıcılardan bir dataframe oluşturuyorum.
final_df = movies_watched_df.loc[movies_watched_df.index.isin(user_same_movies)]
final_df.iloc[:5, :5]

# Oluşturulan bu dataframede her kullanıcının izledikleri filmlere verdikleri puanlarla en yakın kullanıcıları belirlemek için korelasyona bakıyorum.
corr_df = final_df.T.corr().unstack().sort_values().drop_duplicates()
corr_df = pd.DataFrame(corr_df, columns=["corr"])
corr_df.index.names = ["userId_1", "userId_2"]
corr_df = corr_df.reset_index()
corr_df.head()

# Korelasyonları hesapladık.Şimdi de seçtiğim user ile belli bir oranda korelasyona sahip kullanıcıları getireceğim.
top_user = corr_df.loc[(corr_df["userId_1"] == random_user) & (corr_df["corr"] > 0.40)]
top_user = top_user[["userId_2", "corr"]].reset_index(drop=True)
top_user = top_user.sort_values("corr", ascending=False)
top_user = top_user.rename(columns={"userId_2": "userId"})
top_user.head(20)
top_user.shape

# Şu anda top_user dataframe ile random_user ile yüksek korelasyona sahip kullanıcılar elimde.Ama bu kullanıcıların hangi filme kaç puan verdiği bilgisi yok.Onun için rating veri seti ile bu top_user dataframe'ini merge edelim
top_user_ratings = top_user.merge(rating_df[["userId", "movieId", "rating"]], on="userId", how="inner")
top_user_ratings.head(10)
top_user_ratings.shape
top_user_ratings["userId"].nunique()


#############################################
# Adım 5: Weighted Average Recommendation Score'un Hesaplanması
#############################################
# Elimizdeki bu dataframede kullanıcılar, kullanıcıların korelasyon değerleri, film ID'leri ve bu filmlere verilen puanlar var.Burada önermeyi mantıklı bir skor üzerinden yapmalıyım.Sadece korelasyon skorlarına baksam verilen panları atlayacağım.Puanların ortalamalarına göre yapsam bu sefer de her kullanıcının korealsyonu eşit gibi davranacağım.Eğer ki korelasyon ve puanları çarparsam hem puanların hem de korelasyon skorlarının etkisini görebilir ve ona göre önerme yaparım.

top_user_ratings["weighted_rating"] = top_user_ratings["corr"] * top_user_ratings["rating"]
top_user_ratings.head()

recommendation_df = top_user_ratings.groupby("movieId").agg({"weighted_rating": "mean"})
recommendation_df = recommendation_df.reset_index()
recommendation_df.head()

# Belirli bir skor üstü olanları almak istiyorum.
movies_to_be_recommend = recommendation_df.loc[recommendation_df["weighted_rating"] > 1.5]
movies_to_be_recommend = movies_to_be_recommend.reset_index(drop=True)
movies_to_be_recommend.head()

# Film isimlerine erişip kullanıcıya tavsiyede bulunmam için 
movie_end = movies_to_be_recommend.merge(movie_df[["movieId", "title"]], on="movieId", how="inner")
movie_end.head()

# Kullanıcıya Tavsiyede bulunmak için 10 film önereceksek zaten sıralı bir dataframe olduğu için
movie_end.sort_values("weighted_rating", ascending=False)["title"].head(10)

# Tüm bu süreci fonksiyonlaştıralıma.
def user_based_recomender(dataframe, perc_value=65, corr_thresh=0.4, weighted_score=2, rec_thresh=10):
    random_user = dataframe.sample(1, axis=0, random_state=22).index[0]
    random_user_df = dataframe.loc[dataframe.index == random_user]
    user_watch = random_user_df.dropna(axis=1).columns.to_list()
    movies_watch_df = user_movie_df[user_watch]
    user_movie_count = movies_watch_df.notnull().sum(axis=1)
    user_movie_count = pd.DataFrame(user_movie_count, columns=["count"]).reset_index()
    perc = len(user_watch) * perc_value / 100
    user_same_movies = user_movie_count.loc[user_movie_count["count"] > perc, "userId"]
    final_df = movies_watch_df.loc[movies_watch_df.index.isin(user_same_movies)]
    corr_df = final_df.T.corr().unstack().sort_values().drop_duplicates()
    corr_df = pd.DataFrame(corr_df, columns=["corr"])
    corr_df.index.names = ["userId_1", "userId_2"]
    corr_df = corr_df.reset_index()
    top_user = corr_df.loc[(corr_df["userId_1"] == random_user) & (corr_df["corr"] > corr_thresh)]
    top_user = top_user[["userId_2", "corr"]]
    top_user = top_user.reset_index(drop=True).sort_values("corr", ascending=False)
    top_user = top_user.rename(columns={"userId_2": "userId"})
    top_user_ratings = top_user.merge(rating_df[["userId", "movieId", "rating"]], how="inner", on="userId")
    top_user_ratings["weighted_score"] = top_user_ratings["corr"] * top_user_ratings["rating"]
    recemmendation_df = top_user_ratings.groupby("movieId").agg({"weighted_score": "mean"})
    recemmendation_df = recemmendation_df.reset_index()
    movies_to_be_recommend = recemmendation_df.loc[recemmendation_df["weighted_score"] > weighted_score]
    movies_to_be_recommend = movies_to_be_recommend.reset_index(drop=True)
    movie_end = movies_to_be_recommend.merge(movie_df[["movieId", "title"]], how="inner", on="movieId")
    return movie_end.sort_values("weighted_score", ascending=False)["title"].head(10)

user_based_recomender(user_movie_df)









