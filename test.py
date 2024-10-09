from sentence_transformers import SentenceTransformer
sentences = [
    "PKSHA Technologyは機械学習/深層学習技術に関わるアルゴリズムソリューションを展開している。",
    "この深層学習モデルはPKSHA Technologyによって学習され、公開された。",
    "広目天は、仏教における四天王の一尊であり、サンスクリット語の「種々の眼をした者」を名前の由来とする。",
]

#model = SentenceTransformer('pkshatech/simcse-ja-bert-base-clcmlp')
model = SentenceTransformer("cl-nagoya/sup-simcse-ja-base")
embeddings = model.encode(sentences)
print(embeddings)

model = SentenceTransformer('pkshatech/simcse-ja-bert-base-clcmlp')
embeddings = model.encode(sentences)
print(embeddings)
