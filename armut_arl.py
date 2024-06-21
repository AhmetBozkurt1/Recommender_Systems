#########################
# İş Problemi
#########################

# Türkiye’nin en büyük online hizmet platformu olan Armut, hizmet verenler ile hizmet almak isteyenleri buluşturmaktadır.
# Bilgisayarın veya akıllı telefonunun üzerinden birkaç dokunuşla temizlik, tadilat, nakliyat gibi hizmetlere kolayca
# ulaşılmasını sağlamaktadır.
# Hizmet alan kullanıcıları ve bu kullanıcıların almış oldukları servis ve kategorileri içeren veri setini kullanarak
# Association Rule Learning ile ürün tavsiye sistemi oluşturulmak istenmektedir.


#########################
# Veri Seti
#########################
#Veri seti müşterilerin aldıkları servislerden ve bu servislerin kategorilerinden oluşmaktadır.
# Alınan her hizmetin tarih ve saat bilgisini içermektedir.

# UserId: Müşteri numarası
# ServiceId: Her kategoriye ait anonimleştirilmiş servislerdir. (Örnek : Temizlik kategorisi altında koltuk yıkama servisi)
# Bir ServiceId farklı kategoriler altında bulanabilir ve farklı kategoriler altında farklı servisleri ifade eder.
# (Örnek: CategoryId’si 7 ServiceId’si 4 olan hizmet petek temizliği iken CategoryId’si 2 ServiceId’si 4 olan hizmet mobilya montaj)
# CategoryId: Anonimleştirilmiş kategorilerdir. (Örnek : Temizlik, nakliyat, tadilat kategorisi)
# CreateDate: Hizmetin satın alındığı tarih


# Veriyi Hazırlama
import pandas as pd
import datetime as dt
from mlxtend.frequent_patterns import apriori, association_rules
import warnings
warnings.filterwarnings("ignore")

pd.set_option("display.width", 500)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_row", None)

df = pd.read_csv("armut_data.csv")

# Veriye Genel Bakış
def check_df(dataframe, head=5):
    print("####shape####")
    print(dataframe.shape)
    print("####dtype####")
    print(dataframe.dtypes)
    print("####ilk 5 ####")
    print(dataframe.head(head))
    print("####son 5 ####")
    print(dataframe.tail(head))
    print("#### boş değer var mı ####")
    print(dataframe.isnull().sum())
    print("yüzdelik")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)
check_df(df)

# CreateDate değişkeninin dtype değerini "datetime" yapıyoruz.
df["CreateDate"] = df["CreateDate"].apply(lambda x: pd.to_datetime(x))

# ServisID her bir CategoryID özelinde farklı bir hizmeti temsil etmektedir. ServiceID ve CategoryID’yi "_" ile
# birleştirerek bu hizmetleri temsil edecek yeni bir değişken oluşturuyorum.
df["Hizmet"] = df[["ServiceId", "CategoryId"]].apply(lambda x: "_".join(map(str, x)), axis=1)


# Veri seti hizmetlerin alındığı tarih ve saatten oluşmaktadır, herhangi bir sepet tanımı (fatura vb. ) bulunmamaktadır.
# Association Rule Learning uygulayabilmek için bir sepet (fatura vb.) tanımı oluşturulması gerekmektedir. Burada sepet
# tanımı her bir müşterinin aylık aldığı hizmetlerdir. Örneğin; 7256 id'li müşteri 2017'in 8.ayında aldığı 9_4, 46_4
# hizmetleri bir sepeti; 2017’in 10.ayında aldığı 9_4, 38_4 hizmetleri başka bir sepeti ifade etmektedir. Sepetleri
# unique bir ID ile tanımlanması gerekmektedir. Bunun için öncelikle sadece yıl ve ay içeren yeni bir date değişkeni
# oluşturacağım. UserID ve yeni oluşturduğum date değişkenini "_" ile birleştirirek en son ID adında yeni bir değişkene
# atayacağım.

# İlk olarak yıl ve ay bilgisini içeren date değişkenini oluşturacağım.
df["New_Date"] = df["CreateDate"].dt.strftime("%Y-%m")
# Alternatif Yöntem
# df["New_Date"] = df["CreateDate"].apply(lambda x: f"{x.year}-{x.month}")

# Daha sonra UserID ile bu değişkeni birleştiriyorum.
df["SepetId"] = df[["UserId", "New_Date"]].apply(lambda x: "_".join(map(str, x)), axis=1)

#########################
# Birliktelik Kurallarını Üretme
#########################

# Birliktelik kurallarına geçmeden önce veri setimi uygun hale getirmem gerekiyor.0nun için ilk başta Hizmet_Adet diye
# değişken oluşturup her bir değerine 1 verip hizmet sayılarını bulacağım.
df["Hizmet_Adet"] = 1

df_rule = df.groupby(["SepetId", "Hizmet"]).agg({"Hizmet_Adet": "sum"}).reset_index().\
            pivot(index="SepetId", columns="Hizmet", values="Hizmet_Adet")

# NaN değerleri 0 ile dolduruyorum.Ve ardından dataframe de 0 dan büyük olan değerleri 1 yazdıracağız.
df_rule = df_rule.fillna(0)
df_rule = df_rule.applymap(lambda x: 1 if x > 0 else 0)

# Algoritma ve modellere veri setimi hazır hale getirdim.Şimdi de Apriori Algoritması ile sepet birlikteliklerini gösterelim.

frequent_items = apriori(df_rule, min_support=0.01, use_colnames=True)
frequent_items.sort_values("support", ascending=False)

# Association Rule ile birliktelik kurallarını çıkaralım.
frequent_items_arl = association_rules(frequent_items, metric="support", min_threshold=0.01)
sorted_rules = frequent_items_arl.sort_values("confidence", ascending=False)

# Şimdi de seçilen bir ürüne karşılık tavsiyede bulunacağımız fonksiyonu oluşturalım yazalım.
def arl_recommender(dataframe, product_id, resh_count=1):
    recommender_list = []
    for i, product in enumerate(sorted_rules["antecedents"]):
        for j in list(product):
            if j == product_id:
                recommender_list.append(list(sorted_rules["consequents"].iloc[i])[0])
    return recommender_list[0:resh_count]


# Hizmet olarak en son 2_0 hizmetini alan bir kullanıcıya hizmet önerisinde bulunalım.
hizmet_id = "2_0"
arl_recommender(sorted_rules, hizmet_id)
# 15_1 hizmetini tavsiye edebiliriz.

# Birden fazla tavsiyede bulunalım.
hizmet_id = "38_4"
arl_recommender(sorted_rules, hizmet_id, 3)


