import time
import serial
from pynput import keyboard

# >>> AJUSTE A PORTA AQUI <<<
SERIAL_PORT = 'COM3'      # Windows: 'COM3' ... ; Linux: '/dev/ttyUSB0' ; macOS: '/dev/tty.usbserial-xxxx'
BAUD = 115200

ser = serial.Serial(SERIAL_PORT, BAUD, timeout=0)
pressed = set()

print("Controles:")
print("← = L (movimenta mão esquerda)")
print("→ = R (movimenta mão direita)")
print("↑ = C (descansa mão)")
print("0..9 = define o passo (0=1°, 9=10°)")
print("ESC para sair")

# envia comandos contínuos a cada 50 ms enquanto teclas estiverem pressionadas
def pump():
    while True:
        if keyboard.Key.left in pressed:
            ser.write(b'L')
        if keyboard.Key.right in pressed:
            ser.write(b'R')
        time.sleep(0.05)  # 50 ms por passo (ajuste se quiser mais rapido/lento)

def on_press(key):
    if key == keyboard.Key.esc:
        ser.close()
        raise SystemExit

    # setas “one-shot”
    if key == keyboard.Key.up:
        ser.write(b'C')

    # contínuas
    if key in (keyboard.Key.left, keyboard.Key.right):
        pressed.add(key)

    # números mudam o passo no ESP32
    try:
        if hasattr(key, 'char') and key.char is not None and key.char.isdigit():
            ser.write(key.char.encode('ascii'))
    except:
        pass

def on_release(key):
    if key in (keyboard.Key.left, keyboard.Key.right):
        pressed.discard(key)

if __name__ == "__main__":
    # Thread do leitor de teclado
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    try:
        pump()
    except SystemExit:
        pass
    finally:
        if ser and ser.is_open:
            ser.close()