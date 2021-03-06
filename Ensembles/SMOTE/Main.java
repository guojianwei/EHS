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

//
//  Main.java
//
//  Salvador García López
//
//  Created by Salvador García López 30-3-2006.
//  Copyright (c) 2004 __MyCompanyName__. All rights reserved.
//

package keel.Algorithms.ImbalancedClassification.Ensembles.SMOTE;

/**
 * Main.java
 * Main class for the algorithm SMOTE.
 * @author Salvador García López
 */

public class Main {

    /**
     * Main function. Execute the algorithm SMOTE with the configuration file passed as argument.
     * @param args main args. Configuration filename is needed.
     */
    public static void main (String args[]) {

    SMOTE smote;

    if (args.length != 1)
      System.err.println("Error. A parameter is only needed.");
    else {
      smote = new SMOTE (args[0]);
      smote.ejecutar();
    }
  }
}

