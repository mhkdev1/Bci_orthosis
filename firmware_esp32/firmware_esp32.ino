/*
 * firmware_esp32.ino — controle do servo da órtese, lado ESP32.
 *
 * Este é o código gravado DENTRO do ESP32. Do outro lado da serial ficam:
 *   controle_servo_1.py    controle manual por teclado (setas)
 *   inferir_offline.py     inferência em dados gravados, com --ortese
 *   tempo_real.py          inferência ao vivo, com o capacete
 *
 * Os três falam o mesmo protocolo. Nada aqui depende de qual deles está conectado.
 *
 * ---------------------------------------------------------------------------
 * LIGAÇÃO
 * ---------------------------------------------------------------------------
 *   servo (sinal PWM)  ->  GPIO 18
 *   servo (VCC)        ->  fonte externa 5 V, NÃO o 5V do ESP32
 *   servo (GND)        ->  GND comum com o ESP32
 *
 * O GND precisa ser comum, senão o servo treme ou não responde. Alimentar o servo pelo
 * regulador da placa costuma derrubá-la no pico de corrente da partida.
 *
 * ---------------------------------------------------------------------------
 * PROTOCOLO — um byte por comando, sem terminador
 * ---------------------------------------------------------------------------
 *   'L'        angle = max(MIN_ANG, angle - stepDeg)
 *   'R'        angle = min(MAX_ANG, angle + stepDeg)
 *   'C'        angle = 90   (centro do curso; é a posição de repouso da órtese)
 *   '0'..'9'   stepDeg = dígito, com '0' virando 1
 *
 * A cada byte recebido o firmware imprime "angle=<n>". O lado do PC lê esse eco e o usa
 * como posição real, em vez de contar os pulsos às cegas — e precisa lê-lo de qualquer
 * jeito, senão a saída se acumula no buffer do host durante a sessão inteira.
 *
 * ---------------------------------------------------------------------------
 * ATENÇÃO AO CURSO
 * ---------------------------------------------------------------------------
 * MAX_ANG é 180: este firmware NÃO limita o curso à amplitude da órtese. Quem limita é
 * o PC (config.SERVO_AMPLITUDE = 30°, ou seja de 90° a 120°). Isso funciona enquanto o
 * PC for quem comanda, mas não protege contra um terminal aberto por engano nem contra
 * o programa travar com a mão fechada.
 *
 * Para uso com a órtese vestida, considere a variante em firmware_esp32_seguro/, que
 * acrescenta batente por software e retorno automático ao repouso. Ver o README.
 *
 * Dependência: biblioteca "ESP32Servo", pelo Gerenciador de Bibliotecas da IDE Arduino.
 */

#include <ESP32Servo.h>

Servo servo;

const int SERVO_PIN = 18;   // pino do sinal do servo
int angle = 90;             // posição inicial (graus)
int stepDeg = 3;            // incremento por “passo”
const int MIN_ANG = -30;
const int MAX_ANG = 180;

void setup() {
  Serial.begin(115200);
  delay(200);

  // S3003 trabalha a ~50 Hz; limites de pulso típicos 500–2500 µs
  servo.setPeriodHertz(50);
  servo.attach(SERVO_PIN, 500, 2500);
  servo.write(angle);

  Serial.println("Pronto. Comandos: L=esq, R=dir, C=centro, [0-9] muda passo.");
  Serial.print("Angulo: "); Serial.println(angle);
}

void loop() {
  if (Serial.available()) {
    char c = Serial.read();

    if (c == 'L') {                      // seta ESQUERDA
      angle = max(MIN_ANG, angle - stepDeg);
      servo.write(angle);
    } else if (c == 'R') {               // seta DIREITA
      angle = min(MAX_ANG, angle + stepDeg);
      servo.write(angle);
    } else if (c == 'C') {               // seta CIMA (centraliza)
      angle = 90;
      servo.write(angle);
    } else if (c >= '0' && c <= '9') {   // ajustar passo (0=1°, 9=10°)
      stepDeg = (c - '0');
      if (stepDeg == 0) stepDeg = 1;
      if (stepDeg > 10) stepDeg = 10;
      Serial.print("Novo passo: "); Serial.println(stepDeg);
    }

    // feedback no serial
    Serial.print("angle="); Serial.println(angle);
  }
}
