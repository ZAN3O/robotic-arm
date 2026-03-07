/*
 * arduino_arm_control.ino
 * 
 * Firmware for Arduino Mega controlling a 6-servo robotic arm.
 * Receives commands via Serial from Raspberry Pi.
 * 
 * Servo mapping (6 servos):
 *   Pin 8 : Base rotation       (socle)
 *   Pin 5 : Shoulder            (bas - servo 1)
 *   Pin 7 : Elbow               (bas - servo 2)
 *   Pin 6 : Wrist roll          (pince - droite/gauche)
 *   Pin 4 : Wrist yaw           (pince - haut/bas)
 *   Pin 9 : Gripper             (pince - ouvrir/fermer)
 * 
 * Protocol:
 *   Receive: <base,shoulder,elbow,wrist_pitch,wrist_roll,gripper,speed>
 *   Reply:   OK
 *   
 *   All angles are 0-180 degrees.
 *   Speed is 1-100 (delay in ms between steps, mapped inversely).
 */

#include <Servo.h>

// ============================================================
// CONFIGURATION
// ============================================================
#define NUM_SERVOS 6
#define BAUD_RATE 115200

// Servo pins
const int SERVO_PINS[NUM_SERVOS] = {8, 5, 7, 6, 4, 9};

// Servo names (for debug)
const char* SERVO_NAMES[NUM_SERVOS] = {
  "Base", "Shoulder", "Elbow", "WristRoll", "WristYaw", "Gripper"
};

// Home position (degrees)
const int HOME_ANGLES[NUM_SERVOS] = {90, 90, 90, 90, 90, 70};

// Safety limits (min, max) for each servo
const int SERVO_MIN[NUM_SERVOS] = {0,   0,   0,   0,   0,   0};
const int SERVO_MAX[NUM_SERVOS] = {180, 180, 180, 180, 180, 180};

// ============================================================
// GLOBALS
// ============================================================
Servo servos[NUM_SERVOS];
int currentAngles[NUM_SERVOS];
int targetAngles[NUM_SERVOS];

// Serial parsing
const int MAX_CMD_LEN = 128;
char cmdBuffer[MAX_CMD_LEN];
int cmdIndex = 0;
bool receiving = false;

// ============================================================
// SETUP
// ============================================================
void setup() {
  Serial.begin(BAUD_RATE);
  
  // Attach servos
  for (int i = 0; i < NUM_SERVOS; i++) {
    servos[i].attach(SERVO_PINS[i]);
    currentAngles[i] = HOME_ANGLES[i];
    targetAngles[i] = HOME_ANGLES[i];
    servos[i].write(HOME_ANGLES[i]);
  }
  
  delay(500);
  
  Serial.println("READY");
  Serial.print("Servos: ");
  Serial.println(NUM_SERVOS);
}

// ============================================================
// SMOOTH MOVE
// ============================================================
void smoothMove(int speed) {
  // Speed 1=slowest (30ms delay), 100=fastest (1ms delay)
  int delayMs = map(speed, 1, 100, 30, 1);
  delayMs = constrain(delayMs, 1, 30);
  
  // Find max delta to know how many steps
  int maxDelta = 0;
  for (int i = 0; i < NUM_SERVOS; i++) {
    int delta = abs(targetAngles[i] - currentAngles[i]);
    if (delta > maxDelta) maxDelta = delta;
  }
  
  if (maxDelta == 0) return;
  
  // Move in 1-degree increments (synchronized)
  for (int step = 1; step <= maxDelta; step++) {
    for (int i = 0; i < NUM_SERVOS; i++) {
      int delta = targetAngles[i] - currentAngles[i];
      if (delta != 0) {
        // Proportional step
        int pos = currentAngles[i] + (int)((long)delta * step / maxDelta);
        pos = constrain(pos, SERVO_MIN[i], SERVO_MAX[i]);
        servos[i].write(pos);
      }
    }
    delay(delayMs);
  }
  
  // Ensure final position
  for (int i = 0; i < NUM_SERVOS; i++) {
    servos[i].write(targetAngles[i]);
    currentAngles[i] = targetAngles[i];
  }
}

// ============================================================
// PARSE COMMAND
// ============================================================
// Format: <base,shoulder,elbow,wrist_pitch,wrist_roll,gripper,speed>
bool parseCommand(char* cmd) {
  int values[NUM_SERVOS + 1]; // +1 for speed
  int count = 0;
  
  char* token = strtok(cmd, ",");
  while (token != NULL && count < NUM_SERVOS + 1) {
    values[count] = atoi(token);
    count++;
    token = strtok(NULL, ",");
  }
  
  if (count < NUM_SERVOS + 1) {
    Serial.print("ERR:need ");
    Serial.print(NUM_SERVOS + 1);
    Serial.print(" values, got ");
    Serial.println(count);
    return false;
  }
  
  // Set target angles (with safety clamp)
  for (int i = 0; i < NUM_SERVOS; i++) {
    targetAngles[i] = constrain(values[i], SERVO_MIN[i], SERVO_MAX[i]);
  }
  
  int speed = constrain(values[NUM_SERVOS], 1, 100);
  
  // Debug output
  Serial.print("CMD: ");
  for (int i = 0; i < NUM_SERVOS; i++) {
    Serial.print(SERVO_NAMES[i]);
    Serial.print("=");
    Serial.print(targetAngles[i]);
    if (i < NUM_SERVOS - 1) Serial.print(", ");
  }
  Serial.print(" SPD=");
  Serial.println(speed);
  
  // Execute smooth move
  smoothMove(speed);
  
  return true;
}

// ============================================================
// LOOP
// ============================================================
void loop() {
  while (Serial.available() > 0) {
    char c = Serial.read();
    
    if (c == '<') {
      // Start of command
      receiving = true;
      cmdIndex = 0;
      memset(cmdBuffer, 0, MAX_CMD_LEN);
    }
    else if (c == '>') {
      // End of command
      if (receiving) {
        cmdBuffer[cmdIndex] = '\0';
        
        // Check for special commands
        if (strcmp(cmdBuffer, "HOME") == 0) {
          for (int i = 0; i < NUM_SERVOS; i++) {
            targetAngles[i] = HOME_ANGLES[i];
          }
          smoothMove(30);
          Serial.println("OK:HOME");
        }
        else if (strcmp(cmdBuffer, "STOP") == 0) {
          // Emergency stop: just freeze in place
          for (int i = 0; i < NUM_SERVOS; i++) {
            targetAngles[i] = currentAngles[i];
          }
          Serial.println("OK:STOP");
        }
        else if (strcmp(cmdBuffer, "STATUS") == 0) {
          // Report current angles
          Serial.print("POS:");
          for (int i = 0; i < NUM_SERVOS; i++) {
            Serial.print(currentAngles[i]);
            if (i < NUM_SERVOS - 1) Serial.print(",");
          }
          Serial.println();
        }
        else {
          // Parse angle command
          if (parseCommand(cmdBuffer)) {
            Serial.println("OK");
          }
        }
        
        receiving = false;
      }
    }
    else if (receiving) {
      if (cmdIndex < MAX_CMD_LEN - 1) {
        cmdBuffer[cmdIndex++] = c;
      }
    }
  }
}
