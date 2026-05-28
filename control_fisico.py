"""
control_fisico.py
-----------------
Módulo de comunicación UDP con el ESP32 para control físico del péndulo invertido.
El ESP32 crea la red WiFi "PenduloPID" (192.168.4.1).
El PC se conecta a esa red y usa este módulo para:
  - Recibir telemetría: angulo, error, P, I, D, pwm, estado
  - Enviar comandos:    START, STOP, SET Kp Ki Kd tipo
"""

import socket
import threading
import time
import queue
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Configuración de red
# ---------------------------------------------------------------------------
ESP32_IP   = "192.168.4.1"   # IP fija del ESP32 como AP
ESP32_PORT = 4210             # Puerto UDP de escucha del ESP32
PC_PORT    = 4211             # Puerto UDP local para recibir telemetría
TIMEOUT_S  = 2.0              # segundos sin datos → desconectado

# ---------------------------------------------------------------------------
# Estructura de un frame de telemetría
# ---------------------------------------------------------------------------
@dataclass
class FrameTel:
    angulo:  float = 0.0
    error:   float = 0.0
    P:       float = 0.0
    I:       float = 0.0
    D:       float = 0.0
    pwm:     int   = 0
    estado:  str   = "—"
    ts:      float = field(default_factory=time.time)


def parsear_frame(raw: str) -> Optional[FrameTel]:
    """
    Parsea una línea CSV enviada por el ESP32.
    Formato: angulo,error,P,I,D,pwm,estado
    """
    try:
        partes = raw.strip().split(",")
        if len(partes) < 7:
            return None
        return FrameTel(
            angulo  = float(partes[0]),
            error   = float(partes[1]),
            P       = float(partes[2]),
            I       = float(partes[3]),
            D       = float(partes[4]),
            pwm     = int(float(partes[5])),
            estado  = partes[6],
        )
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Cliente UDP — hilo de fondo
# ---------------------------------------------------------------------------
class ClienteESP32:
    """
    Maneja la comunicación UDP con el ESP32 en un hilo separado.
    Thread-safe: usa queue para telemetría y lock para comandos.
    """

    def __init__(self):
        self._sock: Optional[socket.socket] = None
        self._hilo: Optional[threading.Thread] = None
        self._activo = False

        # Cola de frames recibidos (maxsize evita acumulación infinita)
        self.cola: queue.Queue[FrameTel] = queue.Queue(maxsize=500)

        # Último frame recibido (acceso directo sin cola)
        self._ultimo: Optional[FrameTel] = None
        self._lock = threading.Lock()

        # Estadísticas
        self.frames_recibidos = 0
        self.ultimo_rx = 0.0   # timestamp último paquete

    # ------------------------------------------------------------------
    # Conexión / desconexión
    # ------------------------------------------------------------------

    def conectar(self) -> bool:
        """Abre el socket UDP y arranca el hilo receptor."""
        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._sock.bind(("0.0.0.0", PC_PORT))
            self._sock.settimeout(1.0)
            self._activo = True
            self._hilo = threading.Thread(target=self._loop_rx,
                                          daemon=True, name="rx_esp32")
            self._hilo.start()
            return True
        except Exception as e:
            print(f"[ClienteESP32] Error conectando: {e}")
            return False

    def desconectar(self):
        """Cierra el socket y detiene el hilo receptor."""
        self._activo = False
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None

    # ------------------------------------------------------------------
    # Bucle de recepción (hilo de fondo)
    # ------------------------------------------------------------------

    def _loop_rx(self):
        while self._activo:
            try:
                data, _ = self._sock.recvfrom(256)
                raw = data.decode("utf-8", errors="ignore")
                frame = parsear_frame(raw)
                if frame:
                    self.frames_recibidos += 1
                    self.ultimo_rx = time.time()
                    with self._lock:
                        self._ultimo = frame
                    # Encolar sin bloquear (descartar si llena)
                    try:
                        self.cola.put_nowait(frame)
                    except queue.Full:
                        try:
                            self.cola.get_nowait()
                            self.cola.put_nowait(frame)
                        except Exception:
                            pass
            except socket.timeout:
                continue
            except Exception:
                if self._activo:
                    time.sleep(0.1)

    # ------------------------------------------------------------------
    # Lectura de telemetría
    # ------------------------------------------------------------------

    def ultimo_frame(self) -> Optional[FrameTel]:
        """Devuelve el frame más reciente recibido."""
        with self._lock:
            return self._ultimo

    def vaciar_cola(self) -> list:
        """Devuelve todos los frames pendientes en la cola y la vacía."""
        frames = []
        while not self.cola.empty():
            try:
                frames.append(self.cola.get_nowait())
            except queue.Empty:
                break
        return frames

    # ------------------------------------------------------------------
    # Envío de comandos
    # ------------------------------------------------------------------

    def _enviar(self, msg: str):
        """Envía un string al ESP32 por UDP."""
        if not self._sock or not self._activo:
            return
        try:
            self._sock.sendto(msg.encode(), (ESP32_IP, ESP32_PORT))
        except Exception as e:
            print(f"[ClienteESP32] Error enviando: {e}")

    def cmd_start(self):
        """Ordena al ESP32 iniciar el control PID."""
        self._enviar("START")

    def cmd_stop(self):
        """Ordena al ESP32 detener el control."""
        self._enviar("STOP")

    def cmd_set_params(self, tipo: str, Kp: float, Ki: float, Kd: float):
        """
        Envía parámetros del controlador al ESP32.
        Formato: SET,tipo,Kp,Ki,Kd
        Ejemplo: SET,PID,120.0,50.0,8.0
        """
        msg = f"SET,{tipo},{Kp:.3f},{Ki:.3f},{Kd:.3f}"
        self._enviar(msg)

    def cmd_reset_encoder(self):
        """Reinicia el contador del encoder en el ESP32."""
        self._enviar("RESET")

    # ------------------------------------------------------------------
    # Estado de conexión
    # ------------------------------------------------------------------

    @property
    def conectado(self) -> bool:
        """True si el socket está abierto Y se recibió un paquete recientemente."""
        return (self._activo and
                self._sock is not None and
                (time.time() - self.ultimo_rx) < TIMEOUT_S)

    @property
    def segundos_sin_datos(self) -> float:
        if self.ultimo_rx == 0:
            return float("inf")
        return time.time() - self.ultimo_rx
