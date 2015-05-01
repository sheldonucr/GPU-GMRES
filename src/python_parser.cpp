
void python_parser(char* cktname, cs *G, cs *C, cs *B, 
				   Source *VS, int nVS, Source *IS, int nIS, 
				   double tstep, double tstop)
    /* Read G */
	char inGFileName[30];
	strcpy(inGFileName, cktname);
	strcat(inGFileName, ".dataG");
    ifstream inGFile;
    inGFile.open(inGFileName);
	if (!inGFile){
	  cout << "couldn't open " << inGFileName << endl;
	  exit(-1);
	}
    int rowG=0, colG=0, nnzG=0;
    inGFile >> rowG >> colG >> nnzG;
	sparse_mat G(rowG,colG);
    int rg=0, cg=0;
    double valg=0;
    for(int i=0; i<nnzG; i++){
        inGFile >> rg >> cg >> valg;
        G.set_new(rg-1,cg-1,valg);
    }
	G.compact();
	inGFile.close();
  
    /* Read C */
	char inCFileName[30];
	strcpy(inCFileName, cktname);
	strcat(inCFileName, ".dataC");
    ifstream inCFile;
    inCFile.open(inCFileName);
	if (!inCFile){
	  cout << "couldn't open " << inCFileName << endl;
	  exit(-1);
	}
	int rowC=0, colC=0, nnzC=0;
    inCFile >> rowC >> colC >> nnzC;
    sparse_mat C(rowC,colC);
    int rc=0, cc=0;
    double valc=0;
	for(int i=0; i<nnzC; i++){
        inCFile >> rc >> cc >> valc;
        C.set_new(rc-1,cc-1,valc);
    }
	C.compact();
	inCFile.close();
							 
    /* Read B */
	char inBFileName[30];
	strcpy(inBFileName, cktname);
	strcat(inBFileName, ".dataB");
    ifstream inBFile;
    inBFile.open(inBFileName);
	if (!inBFile){
	  cout << "couldn't open " << inBFileName << endl;
	  exit(-1);
	}
    int rowB=0, colB=0, nnzB=0;
    inBFile >> rowB >> colB >> nnzB;
    sparse_mat B(rowB,colB);
    int rb=0, cb=0;
    double valb=0;
    for(int i=0; i<nnzB; i++){
		inBFile >> rb >> cb >> valb;
        B.set_new(rb-1,cb-1,valb);
    }
	B.compact();
	inBFile.close();
	
	/* Read trans */
	double tstep = 0; 
	double tstop = 0;
	char intransFileName[30];
	strcpy(intransFileName, cktname);
	strcat(intransFileName, ".datatrans");
	ifstream intransFile;
	intransFile.open(intransFileName);
	if (!intransFile){
	  cout << "couldn't open " << intransFileName << endl;
	  exit(-1);
	}	
	intransFile >> tstep >> tstop;
	intransFile.close();

	/* Read voltage and current sources */
	Source *VS, *IS;
	int nVS, nIS;
	int nU, cv;
	double tv, vv;
	int pwl_len;
	char str[10];
	char induFileName[30];
	strcpy(induFileName, cktname);
	strcat(induFileName, ".datadu");
	ifstream induFile;
	induFile.open(induFileName);
	if (!induFile){
	  cout << "couldn't open " << induFileName << endl;
	  exit(-1);
	}
	induFile >> nU >> nVS;
	VS = new Source[nVS];
	for(int i = 0; i < nVS; i++){
	  induFile >> cv >> vv;
	  cv--;
	  VS[cv].time.set_size(2);
	  VS[cv].time(0) = 0;
	  VS[cv].time(1) = tstop;
	  VS[cv].value.set_size(2);
	  VS[cv].value(0) = vv;
	  VS[cv].value(1) = vv;
	}
	induFile.close();
	char inpwluFileName[30];
	strcpy(inpwluFileName, cktname);
	strcat(inpwluFileName, ".datapwlu");
	ifstream inpwluFile;
	inpwluFile.open(inpwluFileName);
	if (!inpwluFile){
	  cout << "couldn't open " << inpwluFileName << endl;
	  exit(-1);
	}
	inpwluFile >> nU >> nIS;
	IS = new Source[nIS];
	for(int i = 0; i < nIS; i++){
	  inpwluFile >> str >> cv >> pwl_len;
	  cv -= nVS;
	  cv--;
	  IS[cv].time.set_size(pwl_len);
	  IS[cv].value.set_size(pwl_len);
	  for(int j = 0; j < pwl_len; j++){
		inpwluFile >> tv >> vv;
		IS[cv].time(j) = tv;
		IS[cv].value(j) = vv;
	  }
	}
	induFile.close();

	/* Read ports */
	ivec port;
	int nport;
	char inportFileName[30];
	strcpy(inportFileName, cktname);
	strcat(inportFileName, ".dataport");
	ifstream inportFile;
	inportFile.open(inportFileName);
	if (!inportFile){
	  cout << "couldn't open " << inportFileName << endl;
	  exit(-1);
	}
	inportFile >> nport;
	port.set_size(nport);
	for(int i = 0; i < nport; i++){
	  inportFile >> port(i);
	  port(i)--;
	}
	inportFile.close();

