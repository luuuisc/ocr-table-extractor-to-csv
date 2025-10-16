from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import re

BBOX_RE = re.compile(r"bbox (\d+)\s+(\d+)\s+(\d+)\s+(\d+)")

def parse_bbox(title_attr: str) -> Optional[Tuple[int, int, int, int]]:
    if not title_attr:
        return None
    m = BBOX_RE.search(title_attr)
    if not m:
        return None
    x1, y1, x2, y2 = map(int, m.groups())
    return x1, y1, x2, y2

def overlap_ratio(a1: int, a2: int, b1: int, b2: int) -> float:
    inter = max(0, min(a2, b2) - max(a1, b1))
    denom = max(1, min(a2 - a1, b2 - b1))
    return inter / denom

def within_bbox(bbox: Tuple[int,int,int,int], x1:int,y1:int,x2:int,y2:int) -> bool:
    X1, Y1, X2, Y2 = bbox
    return (x1 >= X1 and y1 >= Y1 and x2 <= X2 and y2 <= Y2)

@dataclass
class Token:
    text: str
    page: int
    x1: int
    y1: int
    x2: int
    y2: int
    line_id: Optional[str] = None

    @property
    def xc(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def yc(self) -> float:
        return (self.y1 + self.y2) / 2.0

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1
