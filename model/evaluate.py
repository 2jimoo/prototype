import json
from collections import defaultdict

"""
MRR (Mean Reciprocal Rank)
- 각 쿼리마다 정답 문서가 처음 등장하는 순위를 계산하고, 그 역수를 구해 누적합니다. 
- 전체 쿼리에 대한 평균을 최종 MRR로 계산합니다.

Forget
- 이전에 성공했던 정답이 이번에는 성공하지 못한 경우를 측정합니다. 
- 이전과 이번의 차이를 비율로 계산해 누적한 뒤 평균을 내어 최종 Forget 값을 구합니다.

FWT (Forward Transfer)
- 새로운 학습이 이전 학습에 긍정적인 영향을 주는지 평가합니다. 
- 이번 학습에서 성공한 정답이 이전 학습에서 실패한 경우, FWT로 측정합니다. 
- 이전과 이번의 차이를 비율로 계산해 누적한 뒤 평균을 내어 최종 FWT 값을 구합니다.
"""


def evaluate_dataset(
    k, query_path, data_path, rankings_path, previous_rankings_path=None
):
    eval_query = set()
    with open(query_path, mode="r") as f:
        for line in f:
            data = json.loads(line)
            qid = int(data["qid"])
            eval_query.add(qid)

    rankings = defaultdict(list)
    with open(rankings_path, "r") as f:
        for line in f:
            items = line.strip().split()
            qid, _, pid, rank, _, _ = items
            qid = int(qid)
            pid = int(pid)
            rank = int(rank)
            if qid in eval_query:
                rankings[qid].append(pid)
                assert rank == len(rankings[qid])

    # If previous rankings are provided, we will also calculate FWT and Forget
    previous_rankings = defaultdict(list)
    if previous_rankings_path:
        with open(previous_rankings_path, "r") as f:
            for line in f:
                items = line.strip().split()
                qid, _, pid, rank, _, _ = items
                qid = int(qid)
                pid = int(pid)
                rank = int(rank)
                if qid in eval_query:
                    previous_rankings[qid].append(pid)
                    assert rank == len(previous_rankings[qid])

    success = 0
    num_q = 0
    recall = 0.0
    mrr = 0.0
    forget = 0.0
    fwt = 0.0

    with open(data_path, mode="r") as f:
        for line in f:
            data = json.loads(line)
            qid = int(data["qid"])
            answer_pids = set(data["answer_pids"])
            if qid in eval_query:
                num_q += 1

                # Success@k and Recall@k
                hit = set(rankings[qid][:k]).intersection(answer_pids)
                if len(hit) > 0:
                    success += 1
                    recall += len(hit) / len(answer_pids)

                # MRR@k
                for i, pid in enumerate(rankings[qid][:k]):
                    if pid in answer_pids:
                        mrr += 1 / (i + 1)
                        break

                # Forget and FWT
                if previous_rankings_path:
                    previous_hit = set(previous_rankings[qid][:k]).intersection(
                        answer_pids
                    )
                    if previous_hit:
                        forget += (len(previous_hit) - len(hit)) / len(answer_pids)
                        fwt += (len(hit) - len(previous_hit)) / len(answer_pids)

    num_rankings = len(rankings)

    print(
        f"# query: {num_rankings}: {num_q}\n",
        f"Success@{k}: {success / num_rankings * 100:.1f}\n",
        f"Recall@{k}: {recall / num_rankings * 100:.1f}\n",
        f"MRR@{k}: {mrr / num_rankings:.4f}\n",
    )

    if previous_rankings_path:
        print(
            f"Forget: {forget / num_rankings * 100:.1f}\n",
            f"FWT: {fwt / num_rankings * 100:.1f}\n",
        )
