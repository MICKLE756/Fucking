import re
from collections import defaultdict


class EntityExtractorRuleBase:
    """实体抽取-基于规则"""

    def __init__(self):
        # 专利号正则 (支持 CN + 数字 + 可选小数点 + 校验位)
        self.patent_id_pattern = re.compile(
            r"CN\d{9,13}[\.\-]?\d?"
        )

        # 约束条件关键词
        self.constraint_keywords = {
            "temperature": {
                "patterns": [
                    r"(\d+)\s*[°℃度]",
                    r"耐[高低]?温\s*(\d+)",
                    r"温度\s*[>≥大于超过不低于]*\s*(\d+)",
                ],
                "keywords": ["耐高温", "高温", "低温", "耐热", "耐温"],
            },
            "application": {
                "patterns": [
                    r"(用于|适用于|应用于|用在)\s*(.+?)[\s，,。]",
                ],
                "keywords": ["餐饮", "厨具", "厨房", "工业", "电子", "航空", "汽车", "医疗", "建筑"],
            },
            "material": {
                "patterns": [],
                "keywords": [
                    "陶瓷", "氟聚合物", "硅基", "碳基", "涂层", "不粘",
                    "纳米", "复合材料", "金属", "有机", "无机",
                ],
            },
            "environmental": {
                "patterns": [],
                "keywords": [
                    "环保", "PFOA", "PFAS", "无毒", "绿色", "可降解",
                    "不含PFOA", "无PFOA", "环境友好",
                ],
            },
            "legal_status": {
                "patterns": [],
                "keywords": ["已授权", "授权", "实质审查", "公开", "有效"],
            },
            "time_range": {
                "patterns": [
                    r"(近|最近)\s*(\d+)\s*(年|月)",
                    r"(\d{4})\s*年?(以[来后]|至今)",
                    r"(\d{4})\s*[-~到至]\s*(\d{4})",
                ],
                "keywords": [],
            },
        }

        # 技术领域词典 (覆盖 milvus.json 中的 tech_field 关键词)
        self.tech_domain_keywords = [
            # 测量/检测
            "测量", "检测", "检具", "传感器", "仪器仪表",
            # 汽车
            "汽车", "车辆", "减振器", "制动", "悬挂", "板簧",
            # 材料
            "涂层", "不粘锅", "耐高温", "纳米材料", "复合材料", "高分子",
            # 制造
            "冲压", "加工", "焊接", "铸造", "模具", "智能制造",
            # 电子/电气
            "电池", "半导体", "光伏", "储能", "新能源", "电磁",
            # 信息技术
            "人工智能", "机器人", "自动驾驶", "物联网", "视觉",
            # 医疗
            "生物医药", "医疗", "基因",
            # 通用
            "环保", "节能", "安全", "通信",
        ]

    def __call__(self, text, schema: list):
        """
        实体抽取，返回 {实体类型: 值}

        schema: 需要抽取的实体类型列表，如 ["技术领域", "核心问题", "约束条件", "专利号"]
        """
        slots = {}

        if "专利号" in schema:
            patent_ids = self.patent_id_pattern.findall(text)
            if patent_ids:
                slots["专利号"] = patent_ids[0]

        if "技术领域" in schema:
            domain = self._extract_tech_domain(text)
            if domain:
                slots["技术领域"] = domain

        if "约束条件" in schema:
            constraints = self._extract_constraints(text)
            if constraints:
                slots["约束条件"] = constraints

        if "核心问题" in schema:
            problem = self._extract_core_problem(text)
            if problem:
                slots["核心问题"] = problem

        return slots

    def _extract_tech_domain(self, text):
        """提取技术领域"""
        found = []
        for keyword in self.tech_domain_keywords:
            if keyword in text:
                found.append(keyword)
        return "、".join(found) if found else None

    def _extract_constraints(self, text):
        """提取约束条件"""
        constraints = {}
        for constraint_type, info in self.constraint_keywords.items():
            # 正则匹配
            for pattern in info["patterns"]:
                match = re.search(pattern, text)
                if match:
                    if constraint_type == "temperature":
                        constraints["temperature_resistance"] = f"{match.group(1)}度以上"
                    elif constraint_type == "application":
                        constraints["application"] = match.group(2).strip()
                    elif constraint_type == "time_range":
                        constraints["time_range"] = match.group(0)
                    break

            # 关键词匹配
            for keyword in info["keywords"]:
                if keyword in text:
                    if constraint_type == "environmental":
                        constraints["environmental_friendly"] = True
                        if "PFOA" in text.upper():
                            constraints["PFOA_free"] = True
                    elif constraint_type == "legal_status":
                        constraints["legal_status"] = keyword
                    elif constraint_type == "application" and "application" not in constraints:
                        constraints["application"] = keyword
                    elif constraint_type == "material":
                        constraints.setdefault("material_type", []).append(keyword)
                    break
        return constraints if constraints else None

    def _extract_core_problem(self, text):
        """提取核心问题"""
        # 匹配"想找/查找/检索 + 内容 + 专利"
        patterns = [
            r"(想找|查找|检索|寻找|搜索)(.*?)(专利|技术|方案)",
            r"(关于|涉及)(.*?)(的专利|的技术)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                problem = match.group(2).strip()
                if problem:
                    return problem
        return None


if __name__ == "__main__":
    extractor = EntityExtractorRuleBase()
    test_texts = [
        ("我想找一种耐高温的不粘锅涂层材料相关的专利", ["技术领域", "核心问题", "约束条件"]),
        ("帮我查一下CN202310001234.5这个专利的详情", ["专利号"]),
        ("主要用于餐饮厨具，需要耐受400度以上高温", ["约束条件"]),
        ("还需要环保，不含PFOA", ["约束条件"]),
        ("只想看已授权的专利", ["约束条件"]),
    ]
    for text, schema in test_texts:
        res = extractor(text, schema)
        print(f"输入: {text}")
        print(f"  Schema: {schema}")
        print(f"  结果: {res}\n")
