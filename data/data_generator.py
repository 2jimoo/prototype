import json
import random


def load_jsonl(file_path, key):
    data = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            data.append(json.loads(line.strip()))
    # data_dict = {item[key]: item for item in data}
    return data


# 쿼리, 문서에 대해서 각각 호출
# 세션에 positive-pair 있다는 보장이 없음(원래 목적에 부합해서 ok)
# 세션당 positive pair 일정 비율 포함 시키는 옵션도 필요한가?
# test set - train에 사용하지 않은 query에 대해 answer_pids가 반환되는가
def drift_sudden(session_length, session_size, domain_a_data, domain_b_data):
    sessions = []
    halfway = session_length // 2

    random.shuffle(domain_a_data)
    random.shuffle(domain_b_data)
    for i in range(halfway):
        start_idx = i * session_size
        end_idx = start_idx + session_size
        session = domain_a_data[start_idx:end_idx]
        sessions.append(session)

    for i in range(halfway, session_length):
        start_idx = (i - halfway) * session_size
        end_idx = start_idx + session_size
        session = domain_b_data[start_idx:end_idx]
        sessions.append(session)

    return sessions


# 쿼리, 문서에 대해서 각각 호출
# 세션에 positive-pair 있다는 보장이 없음(원래 목적에 부합해서 ok)
# 세션당 positive pair 일정 비율 포함 시키는 옵션도 필요한가?
# test set - train에 사용하지 않은 query에 대해 answer_pids가 반환되는가
def drift_gradual(session_length, session_size, domain_a_data, domain_b_data):
    idx_a, idx_b = 0, 0
    sessions = []

    random.shuffle(domain_a_data)
    random.shuffle(domain_b_data)

    for i in range(session_length):
        domain_a_ratio = (session_length - i) / session_length
        num_domain_a = int(session_size * domain_a_ratio)
        num_domain_b = session_size - num_domain_a

        session_a = domain_a_data[idx_a : idx_a + num_domain_a]
        session_b = domain_b_data[idx_b : idx_b + num_domain_b]

        session = session_a + session_b
        random.shuffle(session)

        sessions.append(session)

        idx_a += num_domain_a
        idx_b += num_domain_b

    return sessions


# 쿼리는 어떻게? LLM? document도 이어붙이지 말고 정보 비율 주고 LLM으로?
# docid는 어떻게?
def drift_incremental(session_length, session_size, domain_a_data, domain_b_data):
    sessions = []

    random.shuffle(domain_a_data)
    random.shuffle(domain_b_data)

    idx_a, idx_b, doc_id = 0, 0, 0
    ratio_step = 1 if session_length == 1 else 1 / (session_length - 1)

    for i in range(session_length):
        domain_b_ratio = i * ratio_step if i != session_length - 1 else 1
        domain_a_ratio = 1 - domain_b_ratio

        session = []
        for _ in range(session_size):
            doc_a = domain_a_data[idx_a]["text"]
            doc_b = domain_b_data[idx_b]["text"]
            a_len = len(doc_a)
            b_len = len(doc_b)

            if domain_a_ratio == 0:
                doc_size = b_len
            elif domain_b_ratio == 0:
                doc_size = a_len
            else:
                doc_size = (
                    a_len / domain_a_ratio if a_len < b_len else b_len / domain_b_ratio
                )
            size_domain_a = int(doc_size * domain_a_ratio)
            size_domain_b = int(doc_size * domain_b_ratio)

            print(
                f"a_len:{a_len}, b_len:{a_len} | a_ratio:{domain_a_ratio}, b_ratio:{domain_b_ratio} | doc_size:{doc_size} | a_size:{size_domain_a}, b_size:{size_domain_b}"
            )
            text_a = doc_a[:size_domain_a]
            text_b = doc_b[:size_domain_b]

            text = text_a + text_b
            _session = {
                "id": doc_id,
                "domain_a_doc_id": idx_a,
                "domain_b_doc_id": idx_b,
                "text": text,
            }

            session.append(_session)
            idx_a += 1
            idx_b += 1
            doc_id += 1
        sessions.append(session)

    return sessions


def evolve(partition_length, session_size, domain_a_data, domain_b_data, method):
    if method == "sudden":
        partition1 = drift_sudden(
            partition_length, session_size, domain_a_data, domain_b_data
        )
        partition2 = drift_sudden(
            partition_length, session_size, domain_b_data, domain_a_data
        )
        # session_length/2 Session(domain_a = 100%)
        # session_length/2 Session(domain_b = 100%)
    elif method == "gradual":
        partition1 = drift_gradual(
            partition_length, session_size, domain_a_data, domain_b_data
        )
        partition2 = drift_gradual(
            partition_length, session_size, domain_b_data, domain_a_data
        )
        # Session(domain_a = 100%, domain_b = 0%) -> Session(domain_a = 0%, domain_b = 100%2)
        # Session(domain_a = 0%, domain_b = 100%) -> Session(domain_a = 100%, domain_b = 0%)
    elif method == "incremental":
        partition1 = drift_incremental(
            partition_length, session_size, domain_a_data, domain_b_data
        )
        partition2 = drift_incremental(
            partition_length, session_size, domain_b_data, domain_a_data
        )
        # Doc(domain_a = 100%, domain_b = 0%) -> Doc(domain_a = 0%, domain_b = 100%)
        # Doc(domain_a = 0%, domain_b = 100%) -> Doc(domain_a = 100%, domain_b = 0%)
    return partition1 + partition2
