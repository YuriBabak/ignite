package org.apache.ignite.ml.nn.optimize.listeners;

import org.apache.ignite.ml.nn.api.Model;
import org.apache.ignite.ml.nn.optimize.api.IterationListener;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


public class ScoreIterationListener implements IterationListener {
    private int printIterations = 10;
    private static final Logger log = LoggerFactory.getLogger(ScoreIterationListener.class);
    private boolean invoked = false;
    private long iterCount = 0;

    public ScoreIterationListener(int printIterations) {
        this.printIterations = printIterations;
    }

    @Override
    public void invoke() { this.invoked = true; }

    @Override
    public void iterationDone(Model model, int iteration) {
        if(printIterations <= 0)
            printIterations = 1;
        if(iterCount % printIterations == 0) {
            invoke();
            double result = model.score();
            log.info("Score at iteration " + iterCount + " is " + result);
        }
        iterCount++;
    }
}
