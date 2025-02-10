"""
Inputs: 给你一个matrix里面有customer 的 index 和customer是否enable了coupon, 一个customer， 一个integer kfor exmple：
coupons= {
0: "10% off McDonald's",. check 1point3acres for more.
1: "Free coffee at Starbucks",
2: "20% off cat backpack",
3: "Free soda at taco bell",. .и
4: "$10 for skateboard",
}.
customer_matrix = [
    [1, 1, 1, 0, 0],
    [1, 1, 1, 1, 0],
    [0, 0, 0, 0, 1],. ----
    [1, 0, 1, 1, 0],
    [0, 1, 1, 1, 1],
]
. 1point 3 acres
k = 3
customer = [1,1,1,1,0].
他会给一个similarity的function来判断两个customer是否similar， 要求你找出top k si‍‌‌‍‌‌‍‍‍‍‌‌‌‍‌‌‍milar customers然后给出coupons to recommend。
"""

import heapq

def find_top_k(customer_matrix, customer_input, k, similarity_func):
    heap=[]
    for idx, customer in enumerate(customer_matrix):
        similarity_score=similarity_func(customer,customer_input)
        heapq.heappush(heap, -(similarity_score, idx))
    top_k=[heapq.heappop(heap)[1] for _ in range(min(k, len(heap)))]
    return top_k

def recommendation(customer_matrix, coupons, customer_input, k, similarity_func):
    top_k=find_top_k(customer_matrix, customer_input, k, similarity_func)

    recomnendation=set()
    for idx in top_k:
        for coupon_idx, flag in enumerate(customer_matrix[idx]):
            if flag==1 and customer_input[coupon_idx]==0:
                recommendation.add(coupon_idx)
    return [coupons[coupon_idx] for coupon_idx in recomnendation]


