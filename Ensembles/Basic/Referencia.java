/***********************************************************************

	This file is part of KEEL-software, the Data Mining tool for regression, 
	classification, clustering, pattern mining and so on.

	Copyright (C) 2004-2010
	
	F. Herrera (herrera@decsai.ugr.es)
    L. SÃ¡nchez (luciano@uniovi.es)
    J. AlcalÃ¡-Fdez (jalcala@decsai.ugr.es)
    S. GarcÃ­a (sglopez@ujaen.es)
    A. FernÃ¡ndez (alberto.fernandez@ujaen.es)
    J. Luengo (julianlm@decsai.ugr.es)

	This program is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.

	This program is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with this program.  If not, see http://www.gnu.org/licenses/
  
**********************************************************************/

package keel.Algorithms.ImbalancedClassification.Ensembles.Basic;

public class Referencia
    implements Comparable {

  public int entero;

  public double real;

  public Referencia() {}

  public Referencia(int a, double b) {

    entero = a;

    real = b;

  }

  public int compareTo(Object o1) {

    if (this.real > ( (Referencia) o1).real) {

      return -1;
    }

    else if (this.real < ( (Referencia) o1).real) {

      return 1;
    }

    else {
      return 0;
    }

  }

  public String toString() {

    return new String("{" + entero + ", " + real + "}");

  }

}
