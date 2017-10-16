package keel.Algorithms.ImbalancedClassification.Ensembles.Preprocess.Instance_Selection.EUSCHCQstat;

import keel.Algorithms.ImbalancedClassification.Ensembles.Preprocess.Basic.KNN;
import org.core.*;
import keel.Dataset.*;

/**
 * Abstract(can't be instantiation) Class for minor/major chro
 */
public abstract class BaseChro implements Comparable {

	/*Cromosome data structure*/
	boolean geneBit[];

	/*Useful data for cromosomes*/
	double fitness;
	boolean valid;
	boolean crossed;
	boolean prediction[];
	public BaseChro () {}
	/** 
	 * Construct a  chromosome of specified size
	 * @param size 
	 */
	public BaseChro (int size) {
		int i;
		geneBit = new boolean[size];
		for (i=0; i<size; i++) {
			geneBit[i] = false;
		}
		valid = true;
		crossed = true;
	}


	/**
	 * It returns a given gen of the chromsome
	 * @param indice
	 * @return boolean
	 */
	public boolean getGen (int indice) {
		return geneBit[indice];
	}
	public void setGen(int indice, boolean flag){
		geneBit[indice] = flag;
	}
	public void borrar () {
		valid = false;
	}
	public boolean isValid(){
		return valid;
	}
	/**
	 * Ite returns the fitness of the chrom.
	 * @return
	 */
	public double getFit () {
		return fitness;
	}

	/**
	 * Function that evaluates a cromosome
	 */
	/*
	public abstract void evalua (
			double datos[][], double real[][], int nominal[][], boolean nulos[][],int clases[],
			double train[][], double trainR[][], int trainN[][], boolean trainM[][], int clasesT[],
			BaseChro elite[],
			String wrapper, int K, String evMeas, boolean pFactor, 
			int posID, int nPos, int negID, int nNeg,
			boolean distanceEu, keel.Dataset.Attribute entradas[], 
			boolean[][] anteriores, boolean[][] salidasAnteriores);
	 */
	protected double Qstatistic(boolean[] v1, boolean[] v2, int n) {
		double[][] t = new double[2][2];
		double ceros = 0;
		if (v1.length < n)
			n = v1.length;
		for (int i = 0; i < n; i++) {
			if (v1[i] == v2[i] && v1[i] == true)
				t[0][0]++;
			else if (v1[i] == v2[i] && v1[i] == false)
				t[1][1]++;
			else if (v1[i] != v2[i] && v1[i] == true)
				t[1][0]++;
			else
				t[0][1]++;
			if (!v2[i])
				ceros++;
		}
		if (ceros == n)
			return 2.0;
		return (t[1][1] * t[0][0] - t[0][1] * t[1][0]) / (t[1][1] * t[0][0] + t[0][1] * t[1][0]);
	}
	private double Hamming(boolean[] v1, boolean[] v2, int n){
		int diff = 0;
		for(int i = 0; i<n; i++)
		{
			if(v1[i] != v2[i])
				diff++;
		}
		return (double)diff/(double)n;
	}


	/**
	 * Function that does the CHC diverge
	*/
	public void divergeCHC (double r, Chromosome mejor, double prob) {
		int i;
		for (i=0; i<geneBit.length; i++) {
			if (Randomize.Rand() < r) {
				if (Randomize.Rand() < prob) {
					geneBit[i] = true;
				} else {
					geneBit[i] = false;
				}
			} else {
				geneBit[i] = mejor.getGen(i);
			}
		}
		crossed = true;
	}

	public boolean hasEval () {
		return !crossed;
	}
	/*
	public boolean  () {
		return valido;
	}

	public void borrar () {
		valido = false;
	}
	*/
	public void divergeCHC (double r, BaseChro mejor, double prob) {
		int i;
		for (i=0; i<geneBit.length; i++) {
			if (Randomize.Rand() < r) {
				if (Randomize.Rand() < prob) {
					geneBit[i] = true;
				} else {
					geneBit[i] = false;
				}
			} else {
				geneBit[i] = mejor.getGen(i);
			}
		}
		crossed = true;
	}
	public void setCrossed(boolean flag){
		crossed = flag;
	}
	public int nGenesActivation() {
		int i, sum = 0;
		for (i=0; i<geneBit.length; i++) {
			if (geneBit[i]) sum++;
		}
		return sum;
	}
	
	public int nGenes0Activation (int clases[]) {
		int i, sum = 0;
		for (i=0; i<geneBit.length; i++) {
			if (geneBit[i] && clases[i] == 0) sum++;
		}
		return sum;
	}

	public int nGenes1Activation (int clases[]) {
		int i, sum = 0;
		for (i=0; i<geneBit.length; i++) {
			if (geneBit[i] && clases[i] == 1) sum++;
		}
		return sum;
	}
	
	/**
	 * Prints the chrosome into a string value
	 */
	public int compareTo (Object o1) {
		if (this.getFit() > ((BaseChro)o1).getFit())
			return -1;
		else if (this.getFit() < ((BaseChro)o1).getFit())
			return 1;
		else return 0;
	}

	public String toString() {
		int i;
		String temp = "[";
		for (i=0; i<geneBit.length; i++)
			if (geneBit[i])
				temp += "1";
			else
				temp += "0";
		temp += ", " + String.valueOf(fitness) + ", " + String.valueOf(nGenesActivation()) + "]";
		return temp;
	}
}

