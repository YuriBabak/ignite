package org.apache.ignite.ml.nn.optimize.api;


import org.apache.ignite.ml.nn.api.Model;


public interface IterationListener {
	void invoke();

	void iterationDone(Model model,int iteration);
}
