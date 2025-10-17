import pandas as pd
import unidecode
import re
import pickle

books = pd.read_csv("Books.csv")

ratings = pd.read_csv("Ratings.csv")
mask_shifted = ~books["Year-Of-Publication"].astype(str).str.isnumeric()

for idx in books[mask_shifted].index:
    books.loc[idx, "Image-URL-L"] = books.loc[idx, "Image-URL-M"]
    books.loc[idx, "Image-URL-M"] = books.loc[idx, "Image-URL-S"]
    books.loc[idx, "Image-URL-S"] = books.loc[idx, "Publisher"]
    books.loc[idx, "Publisher"] = books.loc[idx, "Year-Of-Publication"]
    books.loc[idx, "Year-Of-Publication"] = books.loc[idx, "Book-Author"]
    full_title = books.loc[idx, "Book-Title"]
    books.loc[idx, "Book-Title"], books.loc[idx, "Book-Author"] = full_title.split(';')

books["Year-Of-Publication"] = pd.to_numeric(books["Year-Of-Publication"], errors="coerce")
books.loc[(books["Year-Of-Publication"] > 2025), "Year-Of-Publication"] = np.nan

books["Book-Author"] = books["Book-Author"].fillna("Unknown")
books["Publisher"] = books["Publisher"].fillna("Unknown")
books["Year-Of-Publication"] = books["Year-Of-Publication"].fillna(int(books["Year-Of-Publication"].median()))

books = books.drop(columns=["Image-URL-S", "Image-URL-M", "Image-URL-L"])

# убираем лишние пробелы и приводим к нижнему регистру
books["Book-Title"] = books["Book-Title"].astype(str).str.strip().str.lower()
books["Book-Author"] = books["Book-Author"].astype(str).str.strip().str.lower()

# исправляем спецсимволы
books["Book-Title"] = books["Book-Title"].apply(unidecode.unidecode)
books["Book-Author"] = books["Book-Author"].apply(unidecode.unidecode)

# удаляем лишние спецсимволы
def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

books["Book-Title"] = books["Book-Title"].apply(clean_text)
books["Book-Author"] = books["Book-Author"].apply(clean_text)

user_zeros = ratings[ratings["Book-Rating"] == 0].groupby("User-ID").size()
target_user = user_zeros.idxmax()
print("Пользователь для рекомендации:", target_user)

user_zero_books = ratings[(ratings["User-ID"] == target_user) & (ratings["Book-Rating"] == 0)]["ISBN"].tolist()

with open("svd_model.pkl", "rb") as f:
    svd = pickle.load(f)

with open("sgd_model.pkl", "rb") as f:
    linreg_data = pickle.load(f)
    linreg_model = linreg_data["model"]
    target_encoder = linreg_data["target_encoder"]

pred_svd = []
for isbn in user_zero_books:
    try:
        pred = svd.predict(uid=target_user, iid=isbn).est
        if pred >= 8:
            pred_svd.append(isbn)
    except:
        continue

linreg_df = books[books["ISBN"].isin(pred_svd)][["Book-Author", "Publisher", "Year-Of-Publication", "Book-Title"]]

linreg_df_enc = target_encoder.transform(linreg_df)

linreg_pred = linreg_model.predict(linreg_df_enc)
linreg_df = linreg_df.copy()
linreg_df["Pred-Rating"] = linreg_pred

recommendation = linreg_df.sort_values("Pred-Rating", ascending=False)
print("Рекомендованные книги:")
print(recommendation[["Book-Title", "Pred-Rating"]].head(20))