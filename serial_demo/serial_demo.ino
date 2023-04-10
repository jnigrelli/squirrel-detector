int x;
void setup() {
 Serial.begin(115200);
 Serial.setTimeout(1);
}
void loop() {
 while (!Serial.available());
 x = Serial.readString().toInt();
 if (x == 1) {
  Serial.println("Squirrel Detected!");
 } 
}
