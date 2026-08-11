"""Модуль карты: показывает опасное место на Яндекс Карте (JavaScript API).
pywebview должен работать в главном потоке, поэтому окно создаётся в main(),
а камера запускается через webview.start(camera_loop)."""
from pathlib import Path

YANDEX_JS_URL = "https://api-maps.yandex.ru/2.1/?lang=ru_RU&apikey="


def make_map_html(lat: float, lon: float, zoom: int, api_key: str,
                  camera_name: str = "", label: str = "Опасное место") -> str:
    """Возвращает HTML-страницу с Яндекс Картой и красным маркером."""
    return f"""<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8">
<title>Тревожная точка</title>
<style>
  html, body {{ margin:0; padding:0; height:100%; }}
  #map {{ width:100%; height:100%; }}
  .title {{ position:absolute; top:10px; left:10px; z-index:1000;
    background:rgba(220,20,20,.9); color:#fff; padding:8px 14px;
    font:600 15px Arial; border-radius:6px; box-shadow:0 2px 6px rgba(0,0,0,.4); }}
</style></head><body>
<div class="title">{label}{(" &middot; " + camera_name) if camera_name else ""}</div>
<div id="map"></div>
<script src="{YANDEX_JS_URL}{api_key}"></script>
<script>
ymaps.ready(init);
function init() {{
  var map = new ymaps.Map("map", {{
    center: [{lon}, {lat}],
    zoom: {zoom}
  }});
  var placemark = new ymaps.Placemark([{lon}, {lat}], {{
    hintContent: "{label}",
    balloonContent: "{label} ({lat:.5f}, {lon:.5f})"
  }}, {{
    preset: "islands#redDotIcon",
    iconColor: "#d10000"
  }});
  map.geoObjects.add(placemark);
}}
</script>
</body></html>"""


class MapWindow:
    """Окно Яндекс Карты. create() вызывается в главном потоке,
    run(camera_loop) запускает камеру в отдельном потоке и стартует GUI."""

    def __init__(self, config: dict) -> None:
        cfg = config.get("map", {}) or {}
        self.enabled = cfg.get("enabled", True)
        self.api_key = cfg.get("api_key", "")
        self.zoom = cfg.get("zoom", 16)
        self.window_name = cfg.get("window", "Карта: опасное место")
        self.cameras = cfg.get("cameras", [])
        self._html_path = Path("alerts") / "map_trevoga.html"
        self._window = None

    @property
    def usable(self) -> bool:
        return self.enabled and bool(self.api_key)

    def coords_for_camera(self, source: int) -> dict | None:
        """Ищет координаты камеры по индексу source в config['map']['cameras']."""
        for c in self.cameras:
            if c.get("index", 0) == int(source):
                return c
        return None

    def create(self) -> None:
        """Создаёт окно (главный поток). Показываем заглушку, карта загрузится по тревоге."""
        import webview
        self._window = webview.create_window(
            self.window_name,
            html="<body style='background:#eef2f7;font:14px Arial;text-align:center;"
                 "padding-top:80px'>Ожидание тревоги…</body>",
            width=700, height=520,
        )

    def show(self, lat: float, lon: float, camera_name: str = "") -> None:
        """Обновляет содержимое окна карты (можно из любого потока)."""
        if self._window is None:
            return
        html = make_map_html(lat, lon, self.zoom, self.api_key,
                             camera_name=camera_name)
        self._html_path.parent.mkdir(parents=True, exist_ok=True)
        self._html_path.write_text(html, encoding="utf-8")
        try:
            self._window.load_url(str(self._html_path.resolve()))
        except Exception as e:
            print("Не удалось обновить карту:", e)

    def run(self, camera_loop) -> None:
        """Запускает GUI в главном потоке, камеру — в отдельном."""
        import webview
        webview.start(camera_loop)