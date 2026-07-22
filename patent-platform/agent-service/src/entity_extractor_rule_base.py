"""实体抽取 - 基于规则

从用户自然语言中抽取专利检索所需的结构化实体：
    - 专利号
    - 技术领域
    - 核心问题
    - 约束条件 (温度 / 应用场景 / 材料 / 性能指标 / 环保 / 认证标准 /
              法律状态 / 专利类型 / 时间范围 ...)

设计要点：
    - 关键词与正则集中在 ``__init__`` 中声明，便于维护和扩展；
    - 每一类约束由独立的 ``_match_*`` 方法负责，互不影响，新增约束
      只需新增一个方法并在 ``_extract_constraints`` 中登记；
    - 下游 ``workflow_nodes._build_filters`` 依赖 ``application`` 与
      ``time_range`` 两个键，抽取结果保持这两个键名不变以兼容检索过滤。
"""

import re
from typing import Optional


class EntityExtractorRuleBase:
    """基于规则 / 正则的轻量实体抽取器（零外部依赖）。"""

    def __init__(self) -> None:
        # ── 专利号：CN 公开号 / 申请号 (9~13 位数字 + 可选校验位) ──
        self.patent_id_pattern = re.compile(r"CN\d{9,13}[.\-]?[0-9A-Z]?")

        # ── 温度约束 ──
        self.temperature_patterns = [
            r"(-?\d+)\s*[°℃度]?\s*[~\-至到]\s*(-?\d+)\s*[°℃度]",  # -40~150 度 (区间)
            r"(?:零下|负)\s*(\d+)\s*[°℃度]",                   # 零下 40 度
            r"耐[高低]?温\s*[><≥≤]?\s*(\d+)",                  # 耐高温 400
            r"温度\s*[><≥≤大小于超过不低高于至]*\s*(\d+)",      # 温度不低于 200
            r"(\d+)\s*[°℃度]",                                 # 400 度
        ]
        self.temperature_keywords = [
            "耐高温", "高温", "耐热", "耐火", "防火",
            "耐低温", "低温", "耐寒", "耐冷", "极寒", "极地",
            "常温", "恒温", "宽温", "高低温", "冷热冲击",
            "工作温度", "环境温度", "热稳定", "热膨胀",
        ]

        # ── 应用场景 (覆盖语料中的汽车 / 交通 / 工业 / 能源 / 电子 / 民用) ──
        self.application_patterns = [
            r"(?:用于|适用于|应用于|用在|应用在|针对|面向|应用场景为?)\s*(.+?)[\s，,。；;、]",
        ]
        self.application_keywords = [
            # 汽车 / 车辆
            "汽车", "乘用车", "商用车", "新能源汽车", "电动汽车", "燃油车",
            "卡车", "客车", "车载", "整车", "底盘", "车身", "车辆", "动力总成",
            # 轨道 / 航空 / 船舶
            "轨道交通", "高铁", "动车", "列车", "地铁", "船舶", "航空", "航天", "无人机",
            # 工业制造
            "工业", "工厂", "产线", "生产线", "车间", "机床", "装备制造", "重工", "矿山",
            # 能源 / 电力
            "储能", "光伏", "风电", "电网", "充电桩", "动力电池", "电站", "氢能",
            # 电子 / 通信
            "电子", "半导体", "消费电子", "通信", "数据中心", "服务器",
            # 民用 / 其他
            "餐饮", "厨具", "厨房", "家电", "家居", "建筑", "桥梁", "医疗",
            "农业", "户外", "可穿戴", "包装",
        ]

        # ── 材料 ──
        self.material_keywords = [
            # 金属
            "钢", "不锈钢", "高强度钢", "高强钢", "合金钢", "铸铁", "铸钢",
            "铝", "铝合金", "镁合金", "钛", "钛合金", "铜", "黄铜", "合金",
            # 高分子
            "塑料", "工程塑料", "橡胶", "硅胶", "树脂", "尼龙", "聚合物",
            "氟聚合物", "聚四氟乙烯", "高分子", "弹性体",
            # 无机 / 陶瓷 / 碳
            "陶瓷", "玻璃", "石墨", "石墨烯", "硅基", "碳基", "碳纤维",
            "玻璃纤维", "纤维",
            # 涂层 / 复合 / 纳米
            "涂层", "镀层", "不粘", "复合材料", "纳米", "纳米材料",
            # 通用属性
            "金属", "非金属", "有机", "无机", "绝缘材料", "导电材料", "磁性材料",
        ]

        # ── 性能 / 指标约束 ──
        self.performance_patterns = [
            r"(\d+\.?\d*)\s*(MPa|kPa|Pa|bar|兆帕|帕)",              # 压力
            r"(\d+\.?\d*)\s*(kV|V|伏|kW|W|瓦|千瓦|mA|A(?!h)|安)",   # 电气
            r"(?:精度|公差|误差)\s*[±]?\s*(\d+\.?\d*)\s*(μm|um|微米|mm|毫米|%)",  # 精度
            r"(\d+\.?\d*)\s*(mAh|Ah|kWh|Wh)",                       # 容量
            r"(\d+\.?\d*)\s*(rpm|r/min|转|Hz|赫兹)",                # 转速 / 频率
            r"IP\s?(\d{2})",                                        # 防护等级
        ]
        self.performance_keywords = [
            "高精度", "高强度", "高硬度", "耐磨", "耐磨损", "抗冲击", "抗压",
            "耐压", "防水", "防尘", "防腐", "耐腐蚀", "抗腐蚀", "防爆", "阻燃",
            "高效", "节能", "轻量化", "高可靠", "长寿命", "耐久", "密封",
            "减振", "降噪", "智能", "自动化", "小型化", "高灵敏", "快速响应",
            "稳定性", "一致性", "防腐蚀", "抗疲劳", "耐冲击",
        ]

        # ── 环保约束 ──
        self.environmental_keywords = [
            "环保", "无毒", "绿色", "可降解", "可回收", "环境友好",
            "低碳", "碳中和", "节能减排", "清洁", "无污染", "生态",
        ]
        # 环保标准 -> 输出标记键
        self.environmental_standards = {
            "PFOA": "PFOA_free",
            "PFAS": "PFAS_free",
            "ROHS": "RoHS_compliant",
            "REACH": "REACH_compliant",
            "无铅": "lead_free",
            "无卤": "halogen_free",
        }

        # ── 认证 / 标准约束 ──
        self.certification_patterns = [
            r"(GB\s*/?\s*T?\s*\d+(?:\.\d+)*(?:-\d+)?)",  # 国标 GB/T 18384
            r"(ISO\s*\d+)",
            r"(IEC\s*\d+)",
            r"(IATF\s*\d+)",
            r"(SAE\s*[A-Z]?\d+)",
        ]
        self.certification_keywords = [
            "国标", "国家标准", "行业标准", "团体标准", "认证",
            "3C认证", "CE认证", "UL认证", "车规级", "工业级", "军用", "军工",
        ]

        # ── 法律状态约束 ──
        self.legal_status_keywords = [
            "已授权", "授权", "实质审查", "实审", "公开", "公布", "有效",
            "驳回", "撤回", "撤销", "失效", "无效", "届满", "终止", "放弃",
            "PCT", "进入国家阶段", "优先权", "在审",
        ]

        # ── 专利类型约束 ──
        self.patent_type_keywords = {
            "发明专利": "发明专利",
            "发明授权": "发明专利",
            "发明": "发明专利",
            "实用新型": "实用新型",
            "外观设计": "外观设计",
            "外观专利": "外观设计",
        }

        # ── 时间范围约束 ──
        self.time_range_patterns = [
            r"(?:近|最近|过去)\s*(\d+)\s*(?:年|个月|月)",
            r"(\d{4})\s*年?(?:以[来后]|之后|至今)",
            r"(\d{4})\s*[-~到至]\s*(\d{4})",
            r"(\d{4})\s*年",
        ]

        # ── 技术领域词典 (覆盖 milvus.json 中的 tech_field) ──
        self.tech_domain_keywords = [
            # 测量 / 检测 / 工装
            "测量", "检测", "检具", "测具", "量具", "夹具", "工装", "传感器",
            "仪器仪表", "标定", "校准",
            # 汽车 / 传动 / 底盘
            "汽车", "车辆", "减振器", "制动", "刹车", "悬架", "悬挂", "板簧",
            "传动", "变速", "转向", "车桥", "车身", "底盘",
            # 电池 / 电驱 / 电控
            "电池", "动力电池", "电池管理", "BMS", "电机", "电驱", "电控",
            "充电", "储能", "逆变", "电源", "电磁",
            # 材料 / 表面
            "涂层", "不粘锅", "耐高温", "纳米材料", "复合材料", "高分子",
            "表面处理", "热处理", "焊接", "钎焊",
            # 制造工艺
            "冲压", "钣金", "加工", "铸造", "锻造", "注塑", "模具", "智能制造",
            "装配", "焊装", "涂装", "总装",
            # 电子 / 半导体 / 光电
            "半导体", "光伏", "新能源", "光电", "电路", "芯片",
            # 信息技术
            "人工智能", "机器人", "自动驾驶", "物联网", "机器视觉", "视觉",
            "算法", "软件", "数据", "云平台", "导航", "定位", "雷达", "摄像头",
            # 医疗 / 生物
            "生物医药", "医疗", "基因", "诊断",
            # 通用
            "环保", "节能", "安全", "通信", "热管理", "空调", "照明", "密封",
        ]

    # ==================== 主入口 ====================

    def __call__(self, text: str, schema: list) -> dict:
        """按 ``schema`` 抽取实体，返回 ``{实体类型: 值}``。

        Args:
            text: 用户输入文本。
            schema: 需要抽取的实体类型，如
                ``["技术领域", "核心问题", "约束条件", "专利号"]``。
        """
        slots: dict = {}

        if "专利号" in schema:
            patent_id = self._extract_patent_id(text)
            if patent_id:
                slots["专利号"] = patent_id

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

    # ==================== 各实体抽取 ====================

    def _extract_patent_id(self, text: str) -> Optional[str]:
        """抽取首个专利号 (大小写不敏感)。"""
        match = self.patent_id_pattern.search(text.upper())
        return match.group(0) if match else None

    def _extract_tech_domain(self, text: str) -> Optional[str]:
        """抽取技术领域 (去重并保持出现顺序)。"""
        found = []
        for keyword in self.tech_domain_keywords:
            if keyword in text and keyword not in found:
                found.append(keyword)
        return "、".join(found) if found else None

    def _extract_core_problem(self, text: str) -> Optional[str]:
        """抽取核心问题 (检索/查找的对象)。"""
        patterns = [
            r"(?:想找|查找|检索|寻找|搜索|找|查)(.*?)(?:专利|技术|方案|材料)",
            r"(?:关于|涉及|针对)(.*?)(?:的专利|的技术|的方案)",
            r"(?:解决|改善|提升|优化|降低|提高)(.*?)(?:的问题|问题)?$",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                problem = match.group(1).strip(" ，,、")
                if problem:
                    return problem
        return None

    # ==================== 约束条件抽取 ====================

    def _extract_constraints(self, text: str) -> Optional[dict]:
        """抽取所有类别的约束条件。

        每类约束由独立的 ``_match_*`` 方法处理，结果累加进同一个字典。
        保留 ``application`` / ``time_range`` 键名以兼容下游过滤器。
        """
        constraints: dict = {}
        self._match_patent_type(text, constraints)
        self._match_temperature(text, constraints)
        self._match_application(text, constraints)
        self._match_materials(text, constraints)
        self._match_performance(text, constraints)
        self._match_environmental(text, constraints)
        self._match_certification(text, constraints)
        self._match_legal_status(text, constraints)
        self._match_time_range(text, constraints)
        return constraints or None

    def _match_temperature(self, text: str, out: dict) -> None:
        for pattern in self.temperature_patterns:
            match = re.search(pattern, text)
            if match:
                out["temperature"] = re.sub(r"\s+", "", match.group(0))
                break
        for keyword in self.temperature_keywords:
            if keyword in text:
                out.setdefault("temperature_tags", [])
                if keyword not in out["temperature_tags"]:
                    out["temperature_tags"].append(keyword)

    def _match_application(self, text: str, out: dict) -> None:
        for pattern in self.application_patterns:
            match = re.search(pattern, text)
            if match:
                out["application"] = match.group(1).strip()
                break
        if "application" not in out:
            for keyword in self.application_keywords:
                if keyword in text:
                    out["application"] = keyword
                    break

    def _match_materials(self, text: str, out: dict) -> None:
        found = [kw for kw in self.material_keywords if kw in text]
        if found:
            out["material"] = self._dedup(found)

    def _match_performance(self, text: str, out: dict) -> None:
        metrics = []
        for pattern in self.performance_patterns:
            for match in re.finditer(pattern, text):
                metrics.append(match.group(0).strip())
        tags = [kw for kw in self.performance_keywords if kw in text]
        if metrics:
            out["performance_metrics"] = self._dedup(metrics)
        if tags:
            out["performance_tags"] = self._dedup(tags)

    def _match_environmental(self, text: str, out: dict) -> None:
        upper = text.upper()
        if any(kw in text for kw in self.environmental_keywords):
            out["environmental_friendly"] = True
        for token, flag in self.environmental_standards.items():
            if token in upper:
                out[flag] = True

    def _match_certification(self, text: str, out: dict) -> None:
        certs = []
        for pattern in self.certification_patterns:
            for match in re.finditer(pattern, text):
                certs.append(re.sub(r"\s+", "", match.group(1)))
        certs += [kw for kw in self.certification_keywords if kw in text]
        if certs:
            out["certification"] = self._dedup(certs)

    def _match_legal_status(self, text: str, out: dict) -> None:
        for keyword in self.legal_status_keywords:
            if keyword in text:
                out["legal_status"] = keyword
                break

    def _match_patent_type(self, text: str, out: dict) -> None:
        for keyword, canonical in self.patent_type_keywords.items():
            if keyword in text:
                out["patent_type"] = canonical
                break

    def _match_time_range(self, text: str, out: dict) -> None:
        for pattern in self.time_range_patterns:
            match = re.search(pattern, text)
            if match:
                out["time_range"] = match.group(0).strip()
                break

    # ==================== 工具方法 ====================

    @staticmethod
    def _dedup(items: list) -> list:
        """去重并保持顺序。"""
        seen = set()
        result = []
        for item in items:
            if item not in seen:
                seen.add(item)
                result.append(item)
        return result


if __name__ == "__main__":
    extractor = EntityExtractorRuleBase()
    test_texts = [
        ("我想找一种耐高温的不粘锅涂层材料相关的专利", ["技术领域", "核心问题", "约束条件"]),
        ("帮我查一下CN202310001234.5这个专利的详情", ["专利号"]),
        ("主要用于新能源汽车动力电池，需要耐受400度以上高温", ["约束条件"]),
        ("还需要环保，不含PFOA，符合RoHS和GB/T 18384标准", ["约束条件"]),
        ("只想看近3年已授权的发明专利", ["约束条件"]),
        ("要求高强度铝合金，精度0.05mm，防护等级IP67", ["约束条件"]),
        ("找一种工作温度-40~150度的车载传感器", ["技术领域", "约束条件"]),
    ]
    for text, schema in test_texts:
        res = extractor(text, schema)
        print(f"输入: {text}")
        print(f"  Schema: {schema}")
        print(f"  结果: {res}\n")
