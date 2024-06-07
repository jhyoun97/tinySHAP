#include <TensorFlowLite.h>

#include <TensorFlowLite.h>

#include <TensorFlowLite.h>
#include "network_model.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

#define NUMBER_OF_INPUTS 2
#define NUMBER_OF_OUTPUTS 1
#define TENSOR_ARENA_SIZE 100*1024

uint8_t tensor_arena[TENSOR_ARENA_SIZE];
tflite::ErrorReporter* error_reporter;
tflite::MicroInterpreter* interpreter;
TfLiteTensor* input;
TfLiteTensor* output;

int total_samples;
int correct_count;

const float X_test[40][2] = {
    -4.955228546128512, -1.0907715856529432, -20.440108923882583, 
    -1.9957716106741101, -4.9323076001491595, 8.426665643178348, 
    -21.11894200922196, -2.719574275875209, 51.020075722841604, 
    -7.09297055843539, -3.47329447678659, -3.1760334007639144, 
    -14.411442699367562, -2.358050565647164, 2.3547520942415425, 
    9.265458488157034, -12.395531290103705, -1.8574670985079802, 
    -27.91850690787095, -1.2585040122776825, 37.25761294073588, 
    0.8223521178776451, -18.822484616532957, -2.1879644390903983, 
    -26.939319242097444, -1.2319654153574593, -27.848247241155736, 
    -1.567937386910336, -28.433438861976576, -1.7277955508335339, 
    43.99200498682703, -2.8746152456862837, 35.2844392260505, 
    -0.878729349434378, -21.129275799778394, -1.481993850613591, 
    -20.280699812292735, -1.865196982383144, 30.590517681003856, 
    1.9278573040794207, -26.983349902483887, -1.3631112480656056, 
    44.07501705106174, -5.35255310440244, -10.877734774035046, 
    -2.4806562839624386, -6.743901767256732, -3.213675401100972, 
    -22.64973077770228, -2.232137360512732, 44.42927319431864, 
    -3.301637879830898, -21.638230635694782, -1.5897449463485627, 
    -6.196698131843147, 8.960932072898844, -6.360070801097373, 
    -1.632326558411354, 1.699486101583469, 12.151601842601375, 
    13.368332261792252, 15.295260668366097, 23.45891787490427, 
    -0.5126897289041684, -15.08139646959049, -2.6099295856731404, 
    -13.294223658075666, -3.2738866451873876, 35.52752132522992, 
    4.263550532381059, -25.00471825020139, -1.690223673567322, 
    67.30313498311907, -5.210670370484738, -8.86178923356204, 
    -2.066440251611672, 6.135138961598448, 12.338737296262424, 
    -19.70555197642056, -1.5573915995952812
};

const uint8_t y_test[40] = {
    0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 
    0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1
};

void setup() {
    Serial.begin(115200);
    static tflite::MicroErrorReporter micro_error_reporter;
    error_reporter = &micro_error_reporter;

    const tflite::Model* model = tflite::GetModel(network_model);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("Model version does not match schema version.");
        return;
    }

    static tflite::MicroMutableOpResolver<10> micro_op_resolver;
    micro_op_resolver.AddFullyConnected();
    micro_op_resolver.AddSoftmax();
    micro_op_resolver.AddQuantize();
    micro_op_resolver.AddDequantize();

    static tflite::MicroInterpreter static_interpreter(model, micro_op_resolver, tensor_arena, TENSOR_ARENA_SIZE, error_reporter);
    interpreter = &static_interpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("Failed to allocate tensors!");
        return;
    }

    input = interpreter->input(0);
    output = interpreter->output(0);
}

void loop() {
    total_samples = 0;
    correct_count = 0;

    for (uint8_t i = 0; i < 40; i++) {

        total_samples++;

        // Load the i-th test sample data into the input tensor
        for (int j = 0; j < NUMBER_OF_INPUTS; j++) {
            input->data.f[j] = X_test[i][j];
        }

        // Run the model on this input and check for error
        // interpreter->Invoke();
        if (interpreter->Invoke() != kTfLiteOk) {
             Serial.println("Failed to invoke!");
             continue;
        }

        float prediction = output->data.f[0];
        int predicted_class = (prediction > 0.5) ? 0 : 1;

        Serial.print("Sample #");
        Serial.print(i + 1);

        if (predicted_class == 0) {
          Serial.print(", Predicted Class: normal");
        } else {
          Serial.print(", Predicted Class: trigger");
        }

        if (y_test[i] == 0) {
          Serial.println(", Actual Class: normal");
        } else {
          Serial.println(", Actual Class: trigger");
        }

        if (predicted_class == y_test[i]) {
          correct_count++;
        }
        
        // Delay between predictions
        delay(1000);
    }
    Serial.println();
    Serial.println();
    Serial.println("Accuracy Evaluation: ");
    Serial.print("Total number of input samples: ");
    Serial.println(total_samples);
    Serial.print("Total number of samples correctly classified: ");
    Serial.println(correct_count);

    float accuracy = (1.0 * correct_count) /  total_samples;

    Serial.print("Model accuracy: ");
    Serial.println(accuracy);

    Serial.println();
    Serial.println();

    // Delay before repeating the tests
    delay(10000);
}
