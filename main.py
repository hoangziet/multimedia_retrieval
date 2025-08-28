from utils.query import QueryModel

if __name__ == "__main__":
    query_processor = QueryModel()

    kis_query = "Tìm video có chứa một con mèo"
    trake_query = "Tìm 4 khoảnh khắc chính khi vận động viên thực hiện cú nhảy: (1) giậm nhảy, (2) bay qua xà, (3) tiếp đất, (4) đứng dậy."

    results = query_processor.run(query=trake_query, task="trake")
    print(results)
