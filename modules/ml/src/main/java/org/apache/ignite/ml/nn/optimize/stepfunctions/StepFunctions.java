package org.apache.ignite.ml.nn.optimize.stepfunctions;

import org.apache.ignite.ml.nn.optimize.api.StepFunction;


public class StepFunctions {
    private static final NegativeGradientStepFunction NEGATIVE_GRADIENT_STEP_FUNCTION_INSTANCE = new NegativeGradientStepFunction();

    private StepFunctions() {
    }

    public static StepFunction createStepFunction(org.apache.ignite.ml.nn.conf.stepfunctions.StepFunction stepFunction) {
    	if(stepFunction == null ) return null;
        if(stepFunction instanceof org.apache.ignite.ml.nn.conf.stepfunctions.NegativeGradientStepFunction)
            return NEGATIVE_GRADIENT_STEP_FUNCTION_INSTANCE;

        throw new RuntimeException("unknown step function: " + stepFunction);
    }
}
