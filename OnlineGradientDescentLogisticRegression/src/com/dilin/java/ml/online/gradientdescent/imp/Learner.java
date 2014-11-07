package com.dilin.java.ml.online.gradientdescent.imp;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;

import com.dilin.java.ml.online.gradientdescent.util.Calculator;
import com.dilin.java.ml.online.gradientdescent.util.TrainedWrapper;

public class Learner {

	public TrainedWrapper getTrained (String fileName, int lineNum, int dim, double eta) throws NumberFormatException, IOException {
		double[] wAve = new double[dim];
		double[] lambda = new double[dim];
		double[] w = new double[dim];
		double loss = 0.0;
		File file = new File(fileName);
		FileReader fr = new FileReader(file);
		BufferedReader br = new BufferedReader(fr);
		String line = "";
		while((line = br.readLine()) != null && lineNum > 0){
			if(!line.isEmpty()) {
				lineNum--;
				String[] data = line.split(" ");
				int y = Integer.parseInt(data[0]);
				double[] x = new double[dim];
				for(int i = 2; i < data.length; i++){
					String[] keyVal = data[i].split(":");
					if(data.length >= 2) {
						int key =  Integer.parseInt(keyVal[0]);
						double val = Double.parseDouble(keyVal[1]);
						x[key] = val;
					}
				}
				double innerProd = Calculator.getInnerProduct(x, w);
				loss += Math.log(1 + Math.exp(innerProd)) - y * innerProd;
				lambda = Calculator.getLambda(x, innerProd, y);
				for (int j = 0; j < w.length; j++) {
					w[j] = w[j] - eta * lambda[j];
				}

				if (lineNum < 512) {
					for (int j = 0; j < w.length; j++) {
						wAve[j] += w[j];
					}
				}
			}
		}

		for (int j = 0; j < w.length; j++) {
			wAve[j] /= 512;
		}
		br.close();
		return new TrainedWrapper(w, loss, wAve);
	}
	public double getLoss(double[] w, String fileName, int lineNum, int dim, double eta) throws NumberFormatException, IOException {
		double loss = 0.0;
		File file = new File(fileName);
		FileReader fr = new FileReader(file);
		BufferedReader br = new BufferedReader(fr);
		String line = "";
		while((line = br.readLine()) != null && lineNum > 0){
			
			if(!line.isEmpty()) {
				lineNum--;
				String[] data = line.split(" ");
				int y = Integer.parseInt(data[0]);
				double[] x = new double[dim];
				for(int i = 2; i < data.length; i++){
					String[] keyVal = data[i].split(":");
					if(data.length >= 2) {
						int index =  Integer.parseInt(keyVal[0]);
						double val = Double.parseDouble(keyVal[1]);
						x[index] = val;
					}
				}
				double innerProd = Calculator.getInnerProduct(x, w);
				loss += Calculator.getLoss(innerProd, y);
			}
		}
		br.close();
		return loss;
	}
	public int getClassificationError(double[] w, String fileName, int lineNum, int dim, double eta) throws NumberFormatException, IOException {
		int loss = 0;
		File file = new File(fileName);
		FileReader fr = new FileReader(file);
		BufferedReader br = new BufferedReader(fr);
		String line = "";
		while((line = br.readLine()) != null && lineNum > 0){
			
			if(!line.isEmpty()) {
				lineNum--;
				String[] data = line.split(" ");
				int y = Integer.parseInt(data[0]);
				double[] x = new double[dim];
				for(int i = 2; i < data.length; i++){
					String[] keyVal = data[i].split(":");
					if(data.length >= 2) {
						int index =  Integer.parseInt(keyVal[0]);
						double val = Double.parseDouble(keyVal[1]);
						x[index] = val;
					}
				}
				double innerProd = Calculator.getInnerProduct(x, w);
				double expid = 1.0 / (1.0 + Math.exp(-innerProd));
				if (expid > 0.5 && y == 0 || expid <= 0.5 && y == 1) {
					loss++;
				}
			}
		}
		br.close();
		return loss;
	}
	
}
