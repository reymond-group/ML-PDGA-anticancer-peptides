import os
import random
import subprocess as sub
import time

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdchem
from rdkit.Chem import rdmolfiles
from rdkit.Chem import rdmolops

from map4 import MAP4Calculator


def distance_string(a, b):
    """Estimates the Jaccard distance of two binary arrays based on their hashes.

Arguments:
  a {numpy.ndarray} -- An array containing hash values.
  b {numpy.ndarray} -- An array containing hash values.

Returns:
  float -- The estimated Jaccard distance.
"""
    a = set(a)
    b = set(b)
    return 1.0 - (len(a.intersection(b))/len(a.union(b)))


# pop_size, mut_rate, gen_gap, query, sim treshold


def distance(a, b):
    """Estimates the Jaccard distance of two binary arrays based on their hashes.

Arguments:
  a {numpy.ndarray} -- An array containing hash values.
  b {numpy.ndarray} -- An array containing hash values.

Returns:
  float -- The estimated Jaccard distance.
"""
    # The Jaccard distance of Minhashed values is estimated by
    return 1.0 - np.float(np.count_nonzero(a == b)) / np.float(len(a))

# pop_size, mut_rate, gen_gap, query, sim treshold


def cyclize(mol, cy):
    """it is connecting cyclizing the given molecule

    Arguments:
        mol {rdKit mol object} -- molecule to be cyclized
        cy {int} -- 1=yes, 0=no cyclazation

    Returns:
        mols {list of rdKit mol objects} -- possible cyclazation
    """
    count = 0

    # detects all the N terminals in mol
    for atom in mol.GetAtoms():
        if atom.GetSmarts() == '[N:2]' or atom.GetSmarts() == '[NH2:2]' or atom.GetSmarts() == '[NH:2]':
            count += 1
            atom.SetProp('Nterm', 'True')
        else:
            atom.SetProp('Nterm', 'False')

    # detects all the C terminals in mol (it should be one)
    for atom in mol.GetAtoms():
        if atom.GetSmarts() == '[C:1]' or atom.GetSmarts() == '[CH:1]':
            atom.SetProp('Cterm', 'True')
        else:
            atom.SetProp('Cterm', 'False')

    # detects all the S terminals in mol

    for atom in mol.GetAtoms():
        if atom.GetSmarts() == '[S:1]':
            atom.SetProp('Sact1', 'True')
        else:
            atom.SetProp('Sact1', 'False')

    for atom in mol.GetAtoms():
        if atom.GetSmarts() == '[S:2]':
            atom.SetProp('Sact2', 'True')
        else:
            atom.SetProp('Sact2', 'False')

    for atom in mol.GetAtoms():
        if atom.GetSmarts() == '[S:3]':
            atom.SetProp('Sact3', 'True')
        else:
            atom.SetProp('Sact3', 'False')

    Nterm = []
    Cterm = []
    Sact1 = []
    Sact2 = []
    Sact3 = []

    # saves active Cysteins postions:
    for atom in mol.GetAtoms():
        if atom.GetProp('Sact1') == 'True':
            Sact1.append(atom.GetIdx())

    # saves active Cysteins 2 postions:
    for atom in mol.GetAtoms():
        if atom.GetProp('Sact2') == 'True':
            Sact2.append(atom.GetIdx())

    # saves active Cysteins 3 postions:
    for atom in mol.GetAtoms():
        if atom.GetProp('Sact3') == 'True':
            Sact3.append(atom.GetIdx())

    # creates the S-S bond (in the current version only two 'active' Cys, this codo picks two random anyway):
    while len(Sact1) >= 2:
        edmol = rdchem.EditableMol(mol)
        pos = list(range(len(Sact1)))
        x = np.random.choice(pos, 1)[0]
        pos.remove(x)
        y = np.random.choice(pos, 1)[0]
        a = Sact1[x]
        b = Sact1[y]
        edmol.AddBond(a, b, order=Chem.rdchem.BondType.SINGLE)
        mol = edmol.GetMol()
        mol.GetAtomWithIdx(a).SetProp('Sact1', 'False')
        mol.GetAtomWithIdx(b).SetProp('Sact1', 'False')
        mol.GetAtomWithIdx(a).SetAtomMapNum(0)
        mol.GetAtomWithIdx(b).SetAtomMapNum(0)
        Sact1.remove(a)
        Sact1.remove(b)

    while len(Sact2) >= 2:
        edmol = rdchem.EditableMol(mol)
        pos = list(range(len(Sact2)))
        x = np.random.choice(pos, 1)[0]
        pos.remove(x)
        y = np.random.choice(pos, 1)[0]
        a = Sact2[x]
        b = Sact2[y]
        edmol.AddBond(a, b, order=Chem.rdchem.BondType.SINGLE)
        mol = edmol.GetMol()
        mol.GetAtomWithIdx(a).SetProp('Sact2', 'False')
        mol.GetAtomWithIdx(b).SetProp('Sact2', 'False')
        mol.GetAtomWithIdx(a).SetAtomMapNum(0)
        mol.GetAtomWithIdx(b).SetAtomMapNum(0)
        Sact2.remove(a)
        Sact2.remove(b)

    while len(Sact3) >= 2:
        edmol = rdchem.EditableMol(mol)
        pos = list(range(len(Sact3)))
        x = np.random.choice(pos, 1)[0]
        pos.remove(x)
        y = np.random.choice(pos, 1)[0]
        a = Sact3[x]
        b = Sact3[y]
        edmol.AddBond(a, b, order=Chem.rdchem.BondType.SINGLE)
        mol = edmol.GetMol()
        mol.GetAtomWithIdx(a).SetProp('Sact3', 'False')
        mol.GetAtomWithIdx(b).SetProp('Sact3', 'False')
        mol.GetAtomWithIdx(a).SetAtomMapNum(0)
        mol.GetAtomWithIdx(b).SetAtomMapNum(0)
        Sact3.remove(a)
        Sact3.remove(b)

    # saves active C and N terminals postions:
    for atom in mol.GetAtoms():
        if atom.GetProp('Nterm') == 'True':
            Nterm.append(atom.GetIdx())
        if atom.GetProp('Cterm') == 'True':
            Cterm.append(atom.GetIdx())

    if cy == 1:
        edmol = rdchem.EditableMol(mol)

        # creates the amide bond
        edmol.AddBond(Nterm[0], Cterm[0], order=Chem.rdchem.BondType.SINGLE)
        edmol.RemoveAtom(Cterm[0] + 1)

        mol = edmol.GetMol()

        # removes tags and lables form the atoms which reacted
        mol.GetAtomWithIdx(Nterm[0]).SetProp('Nterm', 'False')
        mol.GetAtomWithIdx(Cterm[0]).SetProp('Cterm', 'False')
        mol.GetAtomWithIdx(Nterm[0]).SetAtomMapNum(0)
        mol.GetAtomWithIdx(Cterm[0]).SetAtomMapNum(0)

    return mol


def attach_capping(mol1, mol2):
    """it is connecting all Nterminals with the desired capping

    Arguments:
        mol1 {rdKit mol object} -- first molecule to be connected
        mol2 {rdKit mol object} -- second molecule to be connected - chosen N-capping

    Returns:
        rdKit mol object -- mol1 updated (connected with mol2, one or more)
    """

    count = 0

    # detects all the N terminals in mol1
    for atom in mol1.GetAtoms():
        atom.SetProp('Cterm', 'False')
        if atom.GetSmarts() == '[N:2]' or atom.GetSmarts() == '[NH2:2]' or atom.GetSmarts() == '[NH:2]':
            count += 1
            atom.SetProp('Nterm', 'True')
        else:
            atom.SetProp('Nterm', 'False')

    # detects all the C terminals in mol2 (it should be one)
    for atom in mol2.GetAtoms():
        atom.SetProp('Nterm', 'False')
        if atom.GetSmarts() == '[C:1]' or atom.GetSmarts() == '[CH:1]':
            atom.SetProp('Cterm', 'True')
        else:
            atom.SetProp('Cterm', 'False')

    # mol2 is addes to all the N terminal of mol1
    for i in range(count):
        combo = rdmolops.CombineMols(mol1, mol2)
        Nterm = []
        Cterm = []

        # saves in two different lists the index of the atoms which has to be connected
        for atom in combo.GetAtoms():
            if atom.GetProp('Nterm') == 'True':
                Nterm.append(atom.GetIdx())
            if atom.GetProp('Cterm') == 'True':
                Cterm.append(atom.GetIdx())

        # creates the amide bond
        edcombo = rdchem.EditableMol(combo)
        edcombo.AddBond(Nterm[0], Cterm[0], order=Chem.rdchem.BondType.SINGLE)
        clippedMol = edcombo.GetMol()

        # removes tags and lables form the atoms which reacted
        clippedMol.GetAtomWithIdx(Nterm[0]).SetProp('Nterm', 'False')
        clippedMol.GetAtomWithIdx(Cterm[0]).SetProp('Cterm', 'False')
        clippedMol.GetAtomWithIdx(Nterm[0]).SetAtomMapNum(0)
        clippedMol.GetAtomWithIdx(Cterm[0]).SetAtomMapNum(0)
        # uptades the 'core' molecule
        mol1 = clippedMol

    return mol1



def remove_duplicates(gen):
    """Removes duplicates

    Arguments:
        gen {list} -- sequences

    Returns:
        list -- unique list of sequences
    """

    gen_u = []
    for seq in gen:
        if seq not in gen_u:
            gen_u.append(seq)
    return gen_u


def mating(parents):
    """splits the parents in half and join them giving a child

    Arguments:
        parents {list of strings} -- parents

    Returns:
        string -- child
    """

    parent1 = parents[0]
    parent2 = parents[1]
    half1 = parent1[:random.randint(int(round(len(parent1) / 2, 0)) - 1, int(round(len(parent1) / 2, 0)) + 1)]
    half2 = parent2[random.randint(int(round(len(parent2) / 2, 0)) - 1, int(round(len(parent2) / 2, 0)) + 1):]
    child = half1 + half2

    if 68 in child[1:]:
        print(child)
        child_tmp = child[:1] + child[1:].replace(68,'')
        child = child_tmp
        print(child)
    if 67 in child[:-1]:
        print(child)
        child_tmp = child[:-1].replace(67,'') + child[-1:]
        child = child_tmp
        print(child)
    return child


def swapcy(seq):
    """insertion of two ativated cys at head to tail position

    Arguments:
        seq {string} -- peptide seq

    Returns:
        string -- S-S cyclized peptide seq
    """

    act_cys = 60
    if 60 in seq:
        act_cys = 61
        if 61 in seq:
            act_cys = 62
            if 62 in seq:
                return seq

    new_seq = act_cys + seq[1:] + act_cys

    return new_seq


def break_SS(seq):
    """inactivation of all cys

    Arguments:
        seq {string} -- peptide seq

    Returns:
        string -- S-S cyclized peptide seq
    """

    act_cys = 62
    if 62 not in seq:
        act_cys = 61
        if 61 not in seq:
            act_cys = 60
            if 60 not in seq:
                return seq

    seq.replace(act_cys, '')

    return seq


def set_seed(seed):
    """set seed for random

    Arguments:
        seed {int} -- sed for random
    """

    random.seed(int(seed))
    np.random.seed(int(seed))


class PDGA:
    interprete_dict = {'Arg': 1, 'His': 2, 'Lys': 3, 'Asp': 4, 'Glu': 5, 'Ser': 6, 'Thr': 7, 'Asn': 8,
                       'Gln': 9, 'Cys': 10, 'Gly': 11, 'Pro': 12, 'Ala': 13, 'Ile': 14, 'Leu': 15,
                       'Met': 16, 'Phe': 17, 'Trp': 18, 'Tyr': 19, 'Val': 20, 'Dap': 21, 'Dab': 22,
                       'BOrn': 23, 'BLys': 24, 'Hyp': 29, 'Orn': 30, 'bAla': 31, 'Gaba': 32, 'dDap': 25,
                       'dDab': 26,
                       'dBOrn': 27, 'dBLys': 28, 'dArg': 33, 'dHis': 34, 'dLys': 35, 'dAsp': 36, 'dGlu': 37,
                       'dSer': 38,
                       'dThr': 39, 'dAsn': 40, 'dGln': 41, 'dCys': 42, 'dGly': 43, 'dPro': 44,
                       'dAla': 45,
                       'dIle': 46, 'dLeu': 47, 'dMet': 48, 'dPhe': 49, 'dTrp': 50, 'dTyr': 51, 'dVal': 52,
                       'dHyp': 53, 'dOrn': 54, 'a5a': 55, 'a6a': 56, 'a7a': 57, 'a8a': 58, 'a9a': 59,
                       'Cys1': 60, 'Cys2': 61, 'Cys3': 62, 'dCys1': 63, 'dCys2': 64, 'dCys3': 65,
                       'Ac': 66, 'NH2': 67, 'cy': 68}

    interprete_rev_dict = {v: k for k, v in interprete_dict.items()}

    # list of possible aminoacids
    AA = [1, 2, 3, 5, 6, 7, 8, 9, 11, 12, 13, 20, 14, 15, 17, 19, 18, 10, 4, 16, 29, 30,
          31, 32, 55, 56, 57, 58, 59]
    # list of possible branching units (21=Dap, 22=Dab, 23=Orn, 24=Lys)
    B = [21, 22, 23, 24]
    # list of possible C-terminals
    CT = [67]
    # list of possible N-capping
    NT = [66]

    # variables for random generation of dendrimers
    AA4rndm = [30, 29, 1, 2, 3, 5, 6, 7, 8, 9, 11, 12, 13, 20, 14, 15, 17, 19, 18, 10, 4,
               16,
               31, 32, 55, 56, 57, 58, 59, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    B4rndm = [21, 22, 23, 24, 0]
    CTrndm = [67, 0]
    NTrndm = [66, 0]

    # B4rndm = [0] (only linear generation)
    max_aa_no = 5
    max_gen_no = 3

    # variables for SMILES generation
    B_SMILES = {21: '[N:2][C@@H](C[N:2])[C:1](O)=O', 22: '[N:2][C@@H](CC[N:2])[C:1](O)=O',
                23: '[N:2][C@@H](CCC[N:2])[C:1](O)=O', 24: '[N:2][C@@H](CCCC[N:2])[C:1](O)=O',
                25: '[N:2][C@H](C[N:2])[C:1](O)=O', 26: '[N:2][C@H](CC[N:2])[C:1](O)=O',
                27: '[N:2][C@H](CCC[N:2])[C:1](O)=O', 28: '[N:2][C@H](CCCC[N:2])[C:1](O)=O'}

    AA_SMILES = {13: '[N:2][C@@H](C)[C:1](O)=O', 1: '[N:2][C@@H](CCCNC(N)=N)[C:1](O)=O',
                 8: '[N:2][C@@H](CC(N)=O)[C:1](O)=O', 4: '[N:2][C@@H](CC(O)=O)[C:1](O)=O',
                 10: '[N:2][C@@H](CS)[C:1](O)=O', 9: '[N:2][C@@H](CCC(N)=O)[C:1](O)=O',
                 5: '[N:2][C@@H](CCC(O)=O)[C:1](O)=O', 11: '[N:2]C[C:1](O)=O',
                 2: '[N:2][C@@H](CC1=CNC=N1)[C:1](O)=O', 14: '[N:2][C@@H]([C@@H](C)CC)[C:1](O)=O',
                 3: '[N:2][C@@H](CCCCN)[C:1](O)=O', 15: '[N:2][C@@H](CC(C)C)[C:1](O)=O',
                 16: '[N:2][C@@H](CCSC)[C:1](O)=O', 17: '[N:2][C@@H](CC1=CC=CC=C1)[C:1](O)=O',
                 12: 'C1CC[N:2][C@@H]1[C:1](O)=O', 6: '[N:2][C@@H](CO)[C:1](O)=O',
                 7: '[N:2][C@@H]([C@H](O)C)[C:1](O)=O', 18: '[N:2][C@@H](CC1=CNC2=CC=CC=C12)[C:1](O)=O',
                 19: '[N:2][C@@H](CC1=CC=C(C=C1)O)[C:1](O)=O', 20: '[N:2][C@@H](C(C)C)[C:1](O)=O',
                 60: '[N:2][C@@H](C[S:1])[C:1](O)=O', 61: '[N:2][C@@H](C[S:2])[C:1](O)=O',
                 62: '[N:2][C@@H](C[S:3])[C:1](O)=O',
                 29: 'C1C(O)C[N:2][C@@H]1[C:1](O)=O',
                 30: '[N:2][C@@H](CCCN)[C:1](O)=O', 54: '[N:2][C@H](CCCN)[C:1](O)=O',
                 45: '[N:2][C@H](C)[C:1](O)=O', 33: '[N:2][C@H](CCCNC(N)=N)[C:1](O)=O',
                 40: '[N:2][C@H](CC(N)=O)[C:1](O)=O', 36: '[N:2][C@H](CC(O)=O)[C:1](O)=O',
                 42: '[N:2][C@H](CS)[C:1](O)=O', 41: '[N:2][C@H](CCC(N)=O)[C:1](O)=O',
                 37: '[N:2][C@H](CCC(O)=O)[C:1](O)=O', 43: '[N:2]C[C:1](O)=O',
                 34: '[N:2][C@H](CC1=CNC=N1)[C:1](O)=O', 46: '[N:2][C@H]([C@@H](C)CC)[C:1](O)=O',
                 35: '[N:2][C@H](CCCCN)[C:1](O)=O', 47: '[N:2][C@H](CC(C)C)[C:1](O)=O',
                 48: '[N:2][C@H](CCSC)[C:1](O)=O', 49: '[N:2][C@H](CC1=CC=CC=C1)[C:1](O)=O',
                 44: 'C1CC[N:2][C@H]1[C:1](O)=O', 38: '[N:2][C@H](CO)[C:1](O)=O',
                 39: '[N:2][C@H]([C@H](O)C)[C:1](O)=O', 50: '[N:2][C@H](CC1=CNC2=CC=CC=C12)[C:1](O)=O',
                 51: '[N:2][C@H](CC1=CC=C(C=C1)O)[C:1](O)=O', 52: '[N:2][C@H](C(C)C)[C:1](O)=O',
                 63: '[N:2][C@H](C[S:1])[C:1](O)=O', 64: '[N:2][C@H](C[S:2])[C:1](O)=O',
                 65: '[N:2][C@H](C[S:3])[C:1](O)=O',
                 31: '[N:2]CC[C:1](O)=O', 32: '[N:2]CCC[C:1](O)=O',
                 55: '[N:2]CCCC[C:1](O)=O', 56: '[N:2]CCCCC[C:1](O)=O',
                 57: '[N:2]CCCCCC[C:1](O)=O', 58: '[N:2]CCCCCCC[C:1](O)=O',
                 59: '[N:2]CC[C:1](O)=O'}

    T_SMILES = {67: '[N:2]'}

    C_SMILES = {66: 'C[C:1](=O)'}

    # GA class var
    mut_n = 1
    b_insert_rate = 0.1
    selec_strategy = 'Elitist'
    rndm_newgen_fract = 10

    # initiatization of class variables updated by the GA
    dist_dict_old = {}
    gen_n = 0
    found_identity = 0
    steady_min = 0
    timelimit_seconds = None
    jd_av = None
    jd_min = None
    dist_dict = None
    surv_dict = None
    time = 0
    min_dict = {}

    # can be set with exclude or allow methylation, 
    # it refers to the possibility of having methylation in the entire GA:
    _methyl = False


    # debug
    verbose = False

    def __init__(self, pop_size, mut_rate, gen_gap, query, sim_treshold, porpouse):
        self.MAP4 = MAP4Calculator(dimensions=1024, return_strings=True)
        self.pop_size = int(pop_size)
        self.mut_rate = float(mut_rate)
        self.gen_gap = float(gen_gap)
        self.porpouse = porpouse

        if self.porpouse == 'linear' or self.porpouse == 'cyclic':
            self.B = [0]
            self.B4rndm = [0]
        if not os.path.exists(query):
            os.makedirs(query)
        self.folder = query
        self.query = self.interprete(query)
        self.query_fp = self.calc_map4([self.query])[1][0]
        self.sim_treshold = sim_treshold
        

    def rndm_seq(self):
        """Generates random implicit sequences of max "max_gen_no" generation dendrimers
           with max "max_aa_no" AA in each generation, picking from AA4random, B4random
           (probability of position to be empty intrinsic in these lists). 
        
        Returns:
            string -- implicit sequence of a random dendrimer
        """

        new_random_seq = [random.choice(self.CTrndm)]
        aa_count = 0

        while aa_count < self.max_aa_no:
            new_random_seq.append(random.choice(self.AA4rndm))
            aa_count += 1
        gen_count = 0
        while gen_count < self.max_gen_no:
            new_random_seq.append(random.choice(self.B4rndm))
            aa_count = 0
            while aa_count < self.max_aa_no:
                new_random_seq.append(random.choice(self.AA4rndm))
                aa_count += 1
            gen_count += 1
        new_random_seq.append(random.choice(self.NTrndm))
        new_random_seq = list(filter((1).__ne__, new_random_seq))
        return new_random_seq[::-1]

    def rndm_gen(self):
        """Creates a generation of "pop_size" random dendrimers        
        Returns:
           list -- generation of "pop_size" random dendrimers
        """

        gen = []
        while len(gen) < self.pop_size:
            gen.append(self.rndm_seq())
        return gen

    def find_aa_b_pos(self, seq):
        """finds aminoacids and branching unit positions in a given sequence
        
        Arguments:
            seq {string} -- peptide dendrimer sequence
        
        Returns:
            lists -- aminoacids and branching units positions, all position, terminal pos, capping 
        """

        aa = []
        b = []
        all_pos = []
        met = []

        for i, symbol in enumerate(seq):
            if symbol in [68, 61, 62, 60, 64, 65, 63, 67, 66]:
                continue
            if symbol in self.B_SMILES.keys():
                b.append(i)
            elif symbol in self.AA_SMILES.keys():
                aa.append(i)
            all_pos.append(i)

        return aa, b, met, all_pos

    def split_seq_components(self, seq):
        """split seq in generations and branching units
    
        Arguments:
            seq {string} -- dendrimer sequence

        Returns:
            lists -- generations(gs, from 0 to..), branching units, terminal and capping
        """

        g = []
        gs = []
        bs = []
        t = []
        c = []

        for ix, i in enumerate(seq):
            if i not in [21, 22, 23, 24, 25, 26, 27, 28]:
                if i in self.CT:
                    t.append(i)
                elif i in self.NT:
                    c.append(i)
                elif i == 68:
                    continue
                else:
                    g.append(i)
            else:
                gs.append(g[::-1])
                bs.append(i)
                g = []
        gs.append(g[::-1])
        gs = gs[::-1]
        bs = bs[::-1]

        return gs, bs, t, c

    def pick_aa_b_pos(self, seq, type_pos):
        """If type is aa, it returns an aminoacid position in the given sequence.
        if type is b, it returns a branching unit position in the given sequence.
        if type is all it returns a random position in the given sequence.
        
        Arguments:
            seq {string} -- peptide dendirmer sequence
            type_pos {string} -- aa, b or None
        
        Returns:
            int -- position
        """

        aa_pos, b_pos, met_pos, all_pos = self.find_aa_b_pos(seq)
        try:
            if type_pos == 'aa':
                return random.choice(aa_pos)
            elif type_pos == 'b':
                return random.choice(b_pos)
            elif type_pos == 'met':
                return random.choice(met_pos)
            elif type_pos == 'all':
                return random.choice(all_pos)
            else:
                if self.verbose:
                    print('not valid type, type has to be "aa", "b" or "all"')
        except:
            print(seq)
            
    def connect_mol(self, mol1, mol2):
        """it is connecting all Nterminals of mol1 with the Cterminal 
        of the maximum possible number of mol2s
    
        Arguments:
            mol1 {rdKit mol object} -- first molecule to be connected
            mol2 {rdKit mol object} -- second molecule to be connected

        Returns:
            rdKit mol object -- mol1 updated (connected with mol2, one or more)
        """
        count = 0

        # detects all the N terminals in mol1
        for atom in mol1.GetAtoms():
            atom.SetProp('Cterm', 'False')
            atom.SetProp('methyl', 'False')
            if atom.GetSmarts() == '[N:2]' or atom.GetSmarts() == '[NH2:2]' or atom.GetSmarts() == '[NH:2]':
                count += 1
                atom.SetProp('Nterm', 'True')
            else:
                atom.SetProp('Nterm', 'False')

        # detects all the C terminals in mol2 (it should be one)
        for atom in mol2.GetAtoms():
            atom.SetProp('Nterm', 'False')
            atom.SetProp('methyl', 'False')
            if atom.GetSmarts() == '[C:1]' or atom.GetSmarts() == '[CH:1]':
                atom.SetProp('Cterm', 'True')
            else:
                atom.SetProp('Cterm', 'False')

        # mol2 is addes to all the N terminal of mol1
        for i in range(count):
            combo = rdmolops.CombineMols(mol1, mol2)
            Nterm = []
            Cterm = []

            # saves in two different lists the index of the atoms which has to be connected
            for atom in combo.GetAtoms():
                if atom.GetProp('Nterm') == 'True':
                    Nterm.append(atom.GetIdx())
                if atom.GetProp('Cterm') == 'True':
                    Cterm.append(atom.GetIdx())

            # creates the amide bond
            edcombo = rdchem.EditableMol(combo)
            edcombo.AddBond(Nterm[0], Cterm[0], order=Chem.rdchem.BondType.SINGLE)
            edcombo.RemoveAtom(Cterm[0] + 1)
            clippedMol = edcombo.GetMol()

            # removes tags and lables form c term atoms which reacted
            clippedMol.GetAtomWithIdx(Cterm[0]).SetProp('Cterm', 'False')
            clippedMol.GetAtomWithIdx(Cterm[0]).SetAtomMapNum(0)


            # removes tags and lables form the atoms which reacted
            clippedMol.GetAtomWithIdx(Nterm[0]).SetProp('Nterm', 'False')
            clippedMol.GetAtomWithIdx(Nterm[0]).SetAtomMapNum(0)

            # uptades the 'core' molecule
            mol1 = clippedMol
        
        return mol1

    def smiles_from_seq(self, seq):
        """Calculates the smiles of a given peptide dendrimer sequence
    
        Arguments:
            seq {string} -- peptide dendrimer sequence
        Returns:
            string -- molecule_smile - SMILES of the peptide
        """
        seq = list(filter((0).__ne__,seq))

        gs, bs, terminal, capping = self.split_seq_components(seq)

        # modifies the Cterminal
        if terminal:
            molecule = rdmolfiles.MolFromSmiles(self.T_SMILES[terminal[0]])
        else:
            molecule = ''

        # creates the dendrimer structure
        for gen in gs:
            for aa in gen:
                if molecule == '':
                    molecule = rdmolfiles.MolFromSmiles(self.AA_SMILES[aa])
                else:
                    molecule = self.connect_mol(molecule, rdmolfiles.MolFromSmiles(self.AA_SMILES[aa]))

            if bs:
                if molecule == '':
                    molecule = rdmolfiles.MolFromSmiles(self.B_SMILES[bs[0]])
                else:
                    molecule = self.connect_mol(molecule, rdmolfiles.MolFromSmiles(self.B_SMILES[bs[0]]))
                bs.pop(0)

        # adds capping to the N-terminal (the called clip function is different, cause the listed smiles 
        # for the capping are already without OH, it is not necessary removing any atom after foming the new bond)
        if capping:
            molecule = attach_capping(molecule, rdmolfiles.MolFromSmiles(self.C_SMILES[capping[0]]))

        # clean the smile from all the tags
        for atom in molecule.GetAtoms():
            atom.SetAtomMapNum(0)

        molecule_smile = rdmolfiles.MolToSmiles(molecule, isomericSmiles=True).replace('[N]', 'N').replace('[C]', 'C')
        return molecule_smile

    def smiles_from_seq_cyclic(self, seq):
        """Calculates the smiles of the given peptide sequence and cyclize it
    
        Arguments:
            seq {string} -- peptide dendrimer sequence
        Returns:
            string -- molecule_smile - SMILES of the peptide
        """
        seq = list(filter((0).__ne__,seq))
        
        if 68 in seq:
            cy = 1
            for i in self.NT:
                seq = list(filter((i).__ne__,seq))
            for i in self.CT:
                seq = list(filter((i).__ne__,seq))
        else:
            cy = 0

        gs, bs, terminal, capping = self.split_seq_components(seq)

        # modifies the Cterminal
        if terminal:
            molecule = rdmolfiles.MolFromSmiles(self.T_SMILES[terminal[0]])
        else:
            molecule = ''

        if bs:
            if self.verbose:
                print('dendrimer, cyclization not possible, branching unit will not be considered')

        # creates the linear peptide structure
        for gen in gs:
            for aa in gen:
                if aa == 68:
                    continue
                if molecule == '':
                    molecule = rdmolfiles.MolFromSmiles(self.AA_SMILES[aa])
                else:
                    molecule = self.connect_mol(molecule, rdmolfiles.MolFromSmiles(self.AA_SMILES[aa]))

        # adds capping to the N-terminal (the called clip function is different, cause the listed smiles 
        # for the capping are already without OH, it is not necessary removing any atom after foming the new bond)
        if capping:
            molecule = attach_capping(molecule, rdmolfiles.MolFromSmiles(self.C_SMILES[capping[0]]))

        # cyclize
        if molecule == '':
            smiles = ''
            return smiles, seq
        molecule = cyclize(molecule, cy)

        # clean the smile from all the tags
        for atom in molecule.GetAtoms():
            atom.SetAtomMapNum(0)
        smiles = rdmolfiles.MolToSmiles(molecule, isomericSmiles=True).replace('[N]', 'N').replace('[C]', 'C')

        return smiles, seq
    
    def calc_map4(self, seqs):
        """Calculates the map4 for the given values
        
        Arguments:
            seqs {list} -- peptide sequence list
        
        Returns:
            precessed sesq, fps, smiles, props {lists} -- processed sequences and the relative map4
        """

        proc_seqs = []
        smiles = []
        mol_list = []

        for seq in seqs:
            if seq == []:
                continue
            if self.porpouse == 'cyclic':
                smi, seq = self.smiles_from_seq_cyclic(seq)
                mol = Chem.MolFromSmiles(smi) 
                if mol:
                    smi = Chem.MolToSmiles(mol, isomericSmiles = False)
                    mol = Chem.MolFromSmiles(smi)
                else:
                    print(seq, smi)
                    continue
            else:
                smi = self.smiles_from_seq(seq)
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    smi = Chem.MolToSmiles(mol, isomericSmiles = False)
                    mol = Chem.MolFromSmiles(smi)
                else:
                    print(seq, smi)
                    continue
            if smi == '':
                continue
            
            proc_seqs.append(seq)
            smiles.append(smi)
            mol_list.append(mol)



        fps = self.MAP4.calculate_many(mol_list)
        return proc_seqs, fps, smiles




    def mutate_aa(self, seq):
        """Performs n (mut_n, class variable) random point mutation

        Arguments:
            seq {string} -- seq to be mutated

        Returns:
            list -- mutations
        """
        mutations = []

        for i in range(self.mut_n):
            aa_pos = self.pick_aa_b_pos(seq, 'aa')
            seq1 = seq[:aa_pos]
            seq2 = seq[aa_pos + 1:]
            aa_new = [np.random.choice(self.AA, 1)]
            seq = seq1 + aa_new[0] + seq2
            mutations.append(seq)

        return mutations

    def mutate_b(self, seq):
        """Performs n (mut_n, class variable) random point mutation

        Arguments:
            seq {string} -- seq to be mutated

        Returns:
            list -- mutations
        """
        mutations = []

        for i in range(self.mut_n):
            b_pos = self.pick_aa_b_pos(seq, 'b')
            seq1 = seq[:b_pos]
            seq2 = seq[b_pos + 1:]
            b_new = [np.random.choice(self.B, 1)]
            seq = seq1 + b_new[0] + seq2
            mutations.append(seq)

        return mutations

    def move_b(self, seq, pos=+1):
        """Performs n (mut_n, class variable) random point mutation

        Arguments:
            seq {string} -- seq to be mutated
            pos {integer} -- position to move the branching unit, positive for right, negative for left

        Returns:
            list -- mutations
        """

        mutations = []
        for i in range(self.mut_n):
            b_pos = self.pick_aa_b_pos(seq, 'b')
            b = seq[b_pos]
            if 0 <= b_pos + pos < len(seq):
                if seq[b_pos + pos] in self.CT or seq[b_pos + pos] in self.NT:
                    mutations.append(seq)
                    if self.verbose:
                        print(seq + ' Terminal found, could not move ' + b + ' {}'.format(pos))
                    continue
                else:
                    seqd = seq[:b_pos] + seq[b_pos + 1:]
                    seq1 = seqd[:b_pos + pos]
                    seq2 = seqd[b_pos + pos:]
                    seq = seq1 + [b] + seq2
                    mutations.append(seq)
            else:
                mutations.append(seq)

        return mutations

    def insert(self, seq, type_insert):
        """Performs n (mut_n, class variable) random point insertions. 
        If type insert is 'aa' the new element will be an aminoacid.
        If type insert is 'b' the new element will be a branching unit.
    
        Arguments:
            seq {string} -- seq to be mutated

        Returns:
            list -- mutations
        """

        mutations = []
        for i in range(self.mut_n):
            pos = self.pick_aa_b_pos(seq, 'all')
            if seq[pos] in self.NT or seq[pos] in self.CT or seq[pos] == 68:
                mutations.append(seq)
                continue

            if type_insert == 'aa':
                new_element = np.random.choice(self.AA, 1)
            elif type_insert == 'b':
                new_element = np.random.choice(self.B, 1)
            else:
                raise ValueError("not valid type, type has to be \"aa\" or \"b\"")

            seq1 = seq[:pos]
            seq2 = seq[pos:]
            seq = seq1 + [new_element[0]] + seq2
            mutations.append(seq)

        return mutations

    def delete(self, seq):
        """Performs n (mut_n, class variable) deletion 

        Arguments:
            seq {string} -- seq to be mutated

        Returns:
            list -- mutations
        """

        mutations = []
        for i in range(self.mut_n):
            pos = self.pick_aa_b_pos(seq, 'all')
            seq1 = seq[:pos]
            seq2 = seq[pos + 1:]
            new_seq = seq1 + seq2
            mutations.append(new_seq)

        return mutations

    def fitness_function(self, gen):
        """Calculates the probability of survival of each seq in generation "gen"
    
        Arguments:
            gen {list} -- sequences
            gen_n {int} -- generation number

        Returns:
            jd_av,jd_min {int} -- average and minumum jds of gen
            dist_dict, survival_dict {dict} -- {seq:jd}, {seq:probability_of_survival}
        """

        dist_dict = {}
        gen_to_calc = []

        for seq in gen:
            seq_tuple = tuple(seq)
            if seq_tuple in self.dist_dict_old:
                dist_dict[seq_tuple] = self.dist_dict_old[seq_tuple]
            else:
                gen_to_calc.append(seq)


        seqs, fps, smiles_l = self.calc_map4(gen_to_calc)

        for i, seq in enumerate(seqs):
            map4 = fps[i]
            smiles = smiles_l[i]
            if map4 is None or map4 == '':
                continue
            jd = distance_string(self.query_fp, map4)
            if jd <= self.sim_treshold:
                self.write_results(smiles, seq, map4, jd)
            dist_dict[tuple(seq)] = jd

        survival_dict = {}

        for k, v in dist_dict.items():
            survival_dict[k] = 1 / (v + 1)

        survival_sum = sum(survival_dict.values())
        survival_dict = {k: (v / survival_sum) for k, v in survival_dict.items()}

        jd_av = sum(dist_dict.values()) / len(dist_dict.values())
        jd_min = min(dist_dict.values())

        # updates class variable dist_dict_old
        self.dist_dict_old = dist_dict
        return jd_av, jd_min, dist_dict, survival_dict

    def who_lives(self):
        """Returns the sequences that will remain unchanged
        
        Returns:
            list -- chosen sequences that will live
        """

        sorted_gen = sorted(self.surv_dict.items(), key=lambda x: x[1], reverse=True)
        if sorted_gen[0][0] not in self.min_dict.keys():
            self.min_dict[sorted_gen[0][0]] = self.gen_n
        fraction = int((1 - self.gen_gap) * self.pop_size)
        if len(list(self.surv_dict.keys())) <= fraction:
            wholives_ = []
            wholives = list(self.surv_dict.keys())
            for seq in wholives:
                wholives_.append(list(seq))
            return wholives_

        else:
            wholives = []
            wholives_ = []
            if self.selec_strategy == 'Elitist':
                for element in range(fraction):
                    wholives.append(sorted_gen[element][0])
                for seq in wholives:
                    wholives_.append(list(seq))
                return wholives_
            elif self.selec_strategy == 'Pure':
                while len(wholives) < fraction:
                    new = np.random.choice(list(self.surv_dict.keys()), 1, p=list(self.surv_dict.values()))[0]
                    if new not in wholives:
                        wholives.append(new)
                for seq in wholives:
                    wholives_.append(list(seq))
                return wholives_
            else:
                if self.verbose:
                    print('not valid selection strategy, type has to be "Elitist", or "Pure"')

    def pick_parents(self):
        """Picks two sequences according to their survival probabilities

        Arguments:
            surv_dict {dict} -- {sequence:survival_probability}

        Returns:
            list -- parents
        """

        parents = np.random.choice(list(self.surv_dict.keys()), 2, p=list(self.surv_dict.values()))
        parents_ = [list(parents[0]),list(parents[0])]
        return parents_

    def make_new_gen(self, n):
        """Generates a new generation of n sequences with mating + 2 random sequences

        Arguments:
            n {int} -- number of structure to generate

        Returns:
            list -- new generation
        """

        new_gen = []

        for i in range(int(self.pop_size / self.rndm_newgen_fract)):
            new_gen.append(self.rndm_seq())

        while len(new_gen) < n:
            parents = self.pick_parents()
            child = mating(parents)
            new_gen.append(child)

        return new_gen

    def make_new_gen_cyclic(self, n):
        """Generates a new generation of n sequences with mutation

        Arguments:
            n {int} -- number of structure to generate

        Returns:
            list -- new generation
        """

        new_gen = []
        for i in range(int(self.pop_size / self.rndm_newgen_fract)):
            new_gen.append(self.rndm_seq())

        while len(new_gen) < n / 2:
            parents = self.pick_parents()

            child = mating(parents)

            if 60 in child:
                child = list(filter((60).__ne__,child))
            if 61 in child:
                child = list(filter((61).__ne__,child))
            if 62 in child:
                child = list(filter((62).__ne__,child))

            new_gen.append(child)

        while len(new_gen) < n:
            new_gen.append(np.random.choice(list(self.surv_dict.keys()), 1, p=list(self.surv_dict.values()))[0])
        
        new_gen_ = []
        for seq in new_gen:
            new_gen_.append(list(seq))
        new_gen_cy = self.mutate_cyclic(new_gen_)

        return new_gen_cy

    def mutate(self, gen):
        """Mutates the given generation 

        Arguments:
            gen {list} -- sequences


        Returns:
            [list] -- mutated generation
        """

        mutations = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9']

        mutation = np.random.choice(mutations, 1, replace=False)

        gen_tmp = []

        if mutation == 'M1':
            seq_deletion = np.random.choice(gen, int(round(len(gen) * self.mut_rate, 0)), replace=False)
            for seq in gen:
                if seq in seq_deletion:
                    aa_pos, b_pos, met_pos, all_pos = self.find_aa_b_pos(seq)
                    if all_pos:
                        gen_tmp.append(self.delete(seq)[0])
                    else:
                        gen_tmp.append(seq)
                else:
                    gen_tmp.append(seq)

        if mutation == 'M2':
            seq_insertion_aa = np.random.choice(gen, int(round(len(gen) * self.mut_rate, 0)), replace=False)
            for seq in gen: 
                if seq in seq_insertion_aa:
                    aa_pos, b_pos, met_pos, all_pos = self.find_aa_b_pos(seq)
                    if all_pos:
                        gen_tmp.append(self.insert(seq, 'aa')[0])
                    else:
                        gen_tmp.append(seq)
                else:
                    gen_tmp.append(seq)

        if mutation == 'M3':
            # to avoid incontrolled progressivly growth of the sequences,
            # mantain b_insert_rate (class variable, default = 0.1) low
            seq_insertion_b = np.random.choice(gen, int(round(len(gen) * self.b_insert_rate, 0)), replace=False)
            for seq in gen:
                if seq in seq_insertion_b:
                    aa_pos, b_pos, met_pos, all_pos = self.find_aa_b_pos(seq)
                    if all_pos:
                        gen_tmp.append(self.insert(seq, 'b')[0])                    
                    else:
                        gen_tmp.append(seq)
                else:
                    gen_tmp.append(seq)

        if mutation == 'M4':
            seqs_mutate_aa = np.random.choice(gen, int(round(len(gen) * self.mut_rate, 0)), replace=False)
            for seq in gen:
                if seq in seqs_mutate_aa:
                    aa_pos, b_pos, met_pos, all_pos = self.find_aa_b_pos(seq)
                    if aa_pos:
                        gen_tmp.append(self.mutate_aa(seq)[0])
                    else:
                        gen_tmp.append(seq)
                else:
                    gen_tmp.append(seq)

        if mutation == 'M5':
            seq_move_b_r = np.random.choice(gen, int(round(len(gen) * self.mut_rate, 0)), replace=False)
            for seq in gen:
                if seq in seq_move_b_r:
                    aa_pos, b_pos, met_pos, all_pos = self.find_aa_b_pos(seq)
                    if b_pos:
                        gen_tmp.append(self.move_b(seq, +1)[0])
                    else:
                        gen_tmp.append(seq)
                else:
                    gen_tmp.append(seq)

        if mutation == 'M6':
            seq_move_b_l = np.random.choice(gen, int(round(len(gen) * self.mut_rate, 0)), replace=False)
            for seq in gen:
                if seq in seq_move_b_l:
                    aa_pos, b_pos, met_pos, all_pos = self.find_aa_b_pos(seq)
                    if b_pos:
                        gen_tmp.append(self.move_b(seq, -1)[0])
                    else:
                        gen_tmp.append(seq)
                else:
                    gen_tmp.append(seq)

        if mutation == 'M7':
            seqs_mutate_b = np.random.choice(gen, int(round(len(gen) * self.mut_rate, 0)), replace=False)
            for seq in gen:
                if seq in seqs_mutate_b:
                    aa_pos, b_pos, met_pos, all_pos = self.find_aa_b_pos(seq)
                    if b_pos:
                        gen_tmp.append(self.mutate_b(seq)[0])  
                    else:
                        gen_tmp.append(seq)           
                else:
                    gen_tmp.append(seq)

        if mutation == 'M8':
            seqs_mutate_c = np.random.choice(gen, int(round(len(gen) * self.mut_rate, 0)), replace=False)
            for seq in gen:
                if seq in seqs_mutate_c:
                    c = np.random.choice(self.NT, 1, replace=False)[0]
                    if len(seq) > 2 and seq[0] in self.NT:
                        seq_tmp = seq[1:]
                        new_seq = c + seq_tmp
                        gen_tmp.append(new_seq)
                    else:
                        gen_tmp.append(seq)
                else:
                    gen_tmp.append(seq)

        if mutation == 'M9':
            seqs_mutate_t = np.random.choice(gen, int(round(len(gen) * self.mut_rate, 0)), replace=False)
            for seq in gen:
                if seq in seqs_mutate_t:
                    t = np.random.choice(self.CT, 1, replace=False)[0]
                    if len(seq) > 2 and seq[-1] in self.CT:
                        seq_tmp = seq[:-1]
                        new_seq = seq_tmp + t
                        gen_tmp.append(new_seq)
                    else:
                        gen_tmp.append(seq)
                else:
                    gen_tmp.append(seq)   
        
        if gen_tmp == []:
            gen_tmp = gen
        gen_new = []

        for seq in gen_tmp:
            if seq == '':
                continue
            gen_new.append(seq)

        return gen_new


    def form_SS(self, seq):
        """insertion of two ativated cys
        
        Arguments:
            seq {string} -- peptide seq
        
        Returns:
            string -- S-S cyclized peptide seq
        """

        act_cys = 60
        if 60 in seq:
            act_cys = 61
            if 61 in seq:
                act_cys = 62
                if 62 in seq:
                    return seq

        if len(list(filter((68).__ne__,seq))) <= 2:
            return seq

        # first active cys
        pos = self.pick_aa_b_pos(seq, 'aa')
        seq_tmp = seq[:pos] + act_cys + seq[pos:]

        # second active cys
        pos = self.pick_aa_b_pos(seq, 'aa')
        new_seq = seq_tmp[:pos] + act_cys + seq_tmp[pos:]

        # prevents to activated cys next to each other
        #if act_cys + act_cys not in new_seq:
        seq = new_seq

        return seq

    def mutate_cyclic(self, gen):
        """Mutates the given generation 

        Arguments:
            gen {list} -- sequences

        Returns:
            [list] -- mutated generation
        """

        mutations = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'M10']

        mutation = np.random.choice(mutations, 1, replace=False)

        gen_tmp = []

        if mutation == 'M1':
            seq_deletion = np.random.choice(gen, int(round(len(gen) * self.mut_rate, 0)), replace=False)
            for seq in gen:
                if seq in seq_deletion:
                    aa_pos, b_pos, met_pos, all_pos = self.find_aa_b_pos(seq)
                    if all_pos:
                        new_seq = self.delete(seq)[0]
                        gen_tmp.append(new_seq)
                    else:
                        gen_tmp.append(seq)
                else:
                    gen_tmp.append(seq)

        if mutation == 'M2':
            seq_insertion_aa = np.random.choice(gen, int(round(len(gen) * self.mut_rate, 0)), replace=False)
            for seq in gen:
                if seq in seq_insertion_aa:
                    aa_pos, b_pos, met_pos, all_pos = self.find_aa_b_pos(seq)
                    if all_pos:
                        gen.append(self.insert(seq, 'aa')[0])
                    else:
                        gen_tmp.append(seq)
                else:
                    gen_tmp.append(seq)

        if mutation == 'M3':
            seqs_mutate_aa = np.random.choice(gen, int(round(len(gen) * self.mut_rate, 0)), replace=False)
            for seq in gen:
                if seq in seqs_mutate_aa:
                    aa_pos, b_pos, met_pos, all_pos = self.find_aa_b_pos(seq)
                    if aa_pos:
                        gen_tmp.append(self.mutate_aa(seq)[0])
                    else:
                        gen_tmp.append(seq)
                else:
                    gen_tmp.append(seq)

        if mutation == 'M4':
            seqs_mutate_c = np.random.choice(gen, int(round(len(gen) * self.mut_rate, 0)), replace=False)
            for seq in gen:
                if seq in seqs_mutate_c:
                    if 68 not in seq:
                        c = np.random.choice(self.NT, 1, replace=False)[0]
                        if len(seq) > 2 and seq[0] in self.NT:
                            seq = seq[1:]
                            new_seq = c + seq
                            gen_tmp.append(new_seq)
                        else:
                            gen_tmp.append(seq)
                    else:
                        gen_tmp.append(seq)
                else:
                    gen_tmp.append(seq)
                    
        if mutation == 'M5':
            seqs_mutate_t = np.random.choice(gen, int(round(len(gen) * self.mut_rate, 0)), replace=False)
            for seq in gen:
                if seq in seqs_mutate_t:
                    if 68 not in seq:
                        t = np.random.choice(self.CT, 1, replace=False)[0]
                        if len(seq) > 2 and seq[-1] in self.CT:
                            seq = seq[:-1]
                            new_seq = seq + t
                            gen_tmp.append(new_seq)
                        else:
                            gen_tmp.append(seq)
                    else:
                        gen_tmp.append(seq)
                else:
                    gen_tmp.append(seq)                            

        if mutation == 'M6':
            # break S-S
            seqs_inact_cys = np.random.choice(gen, int(round(len(gen) * self.mut_rate, 0)), replace=False)
            for seq in gen:
                if seq in seqs_inact_cys:
                    gen_tmp.append(break_SS(seq))
                else:
                    gen_tmp.append(seq)

        if mutation == 'M7':
            # make S-S
            seqs_act_cys = np.random.choice(gen, int(round(len(gen) * self.mut_rate, 0)), replace=False)
            for seq in gen:
                if seq in seqs_act_cys:
                    act_seq = self.form_SS(seq)
                    gen_tmp.append(act_seq)
                else:
                    gen_tmp.append(seq)

        if mutation == 'M8':
            # linearize
            seqs_lin = np.random.choice(gen, int(round(len(gen) * self.mut_rate, 0)), replace=False)
            for seq in gen:
                if seq in seqs_lin:
                    if 68 in seq:
                        new_seq = seq[1:]
                        gen_tmp.append(new_seq)
                    else:
                        gen_tmp.append(seq)
                else:
                    gen_tmp.append(seq)

        if mutation == 'M9':
            # cyclize
            seqs_cy = np.random.choice(gen, int(round(len(gen) * self.mut_rate, 0)), replace=False)
            for seq in gen:
                if seq in seqs_cy:
                    if 68 not in seq:
                        new_seq = 68 + seq
                        for i in self.NT:
                            new_seq = new_seq.replace(i, '')
                        for i in self.CT:
                            new_seq = new_seq.replace(i, '')
                        gen_tmp.append(new_seq)
                    else:
                        gen_tmp.append(seq)
                else:
                    gen_tmp.append(seq)

        if mutation == 'M10':
            # swap head-to-tail with S-S
            seqs_swap = np.random.choice(gen, int(round(len(gen) * self.mut_rate, 0)), replace=False)
            for seq in gen:
                if seq in seqs_swap:
                    if 68 in seq:
                        new_seq = swapcy(seq)
                        gen_tmp.append(new_seq)
                    else:
                        gen_tmp.append(seq)
                else:
                    gen_tmp.append(seq)

        if gen_tmp == []:
            gen_tmp = gen
        gen_new = []

        for seq in gen_tmp:
            gen_new.append(seq)

        #for seq in gen_tmp:
        #    if seq == 68 or seq == '':
        #        continue
        #    else:
        #        gen_new.append(seq)

        #for seq in gen_tmp:
        #    for i in ['ÄÄ', 'ÖÖ', 'ÜÜ']:
        #        if i in seq:
        #            new_seq = seq.replace(i, '')
        #            gen_new.append(new_seq)
        #        else:
        #            gen_new.append(seq)

        #for seq in gen_tmp:
        #    for i in self.NT:
        #        if i in seq and 68 in seq:
        #            new_seq = seq.replace(i, '')
        #            gen_new.append(new_seq)
        #        else:
        #            gen_new.append(seq)

        #for seq in gen_tmp:
        #    for i in self.CT:
        #        if i in seq and 68 in seq:
        #            new_seq = seq.replace(i, '')
        #            gen_new.append(new_seq)
        #        else:
        #            gen_new.append(seq)
        
        #if self._methyl:
        #    for seq in gen_tmp:
        #        if '--' in seq:
        #            new_seq = seq.replace('--', '-')
        #            gen_new.append(new_seq)
        #        else:
        #            gen_new.append(seq)
        
        return gen_new

    def write_results(self, smiles, seq, map4, jd):
        """if jd from query is smaller than similarity treshold
            (class variable), adds seq to results
        
        """
        with open('{}/{}_results'.format(self.folder, self.reinterprete(self.query)), 'a') as outFile:
            outFile.write(smiles + ' ' + self.reinterprete(seq) + ' ' + str(round(jd,3)) + '\n')

    def write_progress(self):
        """add gen number, gen sequences and its jd av and min
        
        """

        gen_temp = []
        gen = list(self.dist_dict.keys())
        for seq in gen:
            seq = list(seq)
            gen_temp.append(self.reinterprete(seq))
        gen = ';'.join(map(str, gen_temp))
        with open('{}/{}_generations'.format(self.folder, self.reinterprete(self.query)), 'a') as outFile:
            outFile.write(str(self.gen_n) + ' ' + gen + ' ' + str(self.jd_av) + ' ' + str(self.jd_min) + '\n')

    def write_param(self):
        with open('{}/param.txt'.format(self.folder), '+w') as outFile:
            outFile.write(str(self.__dict__) + '\n')
            outFile.write('Class variables: ' + '\n')
            outFile.write('used AA: ' + str(self.AA) + '\n')
            outFile.write('number of point mutation: ' + str(self.mut_n) + '\n')
            outFile.write('insert branching unit rate (mutation): ' + str(self.b_insert_rate) + '\n')
            outFile.write('survival strategy: ' + str(self.selec_strategy) + '\n')
            outFile.write('fraction of new generation that is random: ' + str(self.rndm_newgen_fract) + '\n')

    def set_verbose_true(self):
        """set verbose true (default false)
        """

        self.verbose = True

    def set_verbose_false(self):
        """set verbose false (default false)
        """

        self.verbose = False

    def exclude_buildingblocks(self, bb_to_ex):
        """Excludes the given building blocks
        
        Arguments:
            bb_to_ex {list} -- building blocks to exclude
        """

        for bb in bb_to_ex:
            if bb in self.interprete_dict.keys():
                element = self.interprete(bb)
                if element in self.AA:
                    self.exclude_aminoacids(element)
                elif element in self.B:
                    self.exclude_branching(element)
                elif element in self.CT:
                    self.exclude_C_terminal(element)
                elif element in self.NT:
                    self.exclude_N_capping(element)                
                elif bb == 'met':
                    self.exclude_methylation()
                else:
                    print("can't exclude ", bb)
            else:
                print("can't exclude ", bb)


    def exclude_aminoacids(self, aa_to_ex):
        """Excludes the given aminoacids
        
        Arguments:
            aa_to_ex {list} -- aminoacids to exclude
        """

        for element in aa_to_ex:
            self.AA.remove(element)
            self.AA4rndm.remove(element)
            self.AA4rndm.remove(0)

        if not self.AA:
            self.AA.append(0)
        if not self.AA4rndm:
            self.AA4rndm.append(0)

        if self.verbose:
            print('The GA is using aminoacids:', self.AA)

    def exclude_branching(self, bb_to_ex):
        """Excludes the given branching units
        
        Arguments:
            bb_to_ex {list} -- branching units to exclude
        """

        if self.porpouse != 'cyclic':

            for element in bb_to_ex:
                self.B.remove(element)
                self.B4rndm.remove(element)

            if not self.B:
                self.B.append(0)
            if not self.B4rndm:
                self.B4rndm.append(0)

            if self.verbose:
                print('The GA is using branching units:', self.B)

    def exclude_methylation(self):
        """excludes the possibility of amide bond methylation
        """
        self._methyl = False

    def allow_methylation(self):
        """allows the possibility of amide bond methylation
        """
        self._methyl = True

    def exclude_C_terminal(self, t_to_ex):
        """Excludes the given C terminal modifications
        
        Arguments:
            t_to_ex {list} -- C terminal modifications to exclude
        """

        for element in t_to_ex:
            self.CT.remove(element)
            self.CTrndm.remove(element)
            self.CTrndm.remove(0)

        if not self.CT:
            self.CT.append(0)
        if not self.CTrndm:
            self.CTrndm.append(0)

        if self.verbose:
            print('The GA is using C terminal mod:', self.CT)

    def exclude_N_capping(self, c_to_ex):
        """Excludes the given N terminal capping
        
        Arguments:
            c_to_ex {list} -- N terminal capping to exclude
        """

        for element in c_to_ex:
            self.NT.remove(element)
            self.NTrndm.remove(element)
            self.NTrndm.remove(0)

        if not self.NT:
            self.NT.append(0)
        if not self.NTrndm:
            self.NTrndm.append(0)

        if self.verbose:
            print('The GA is using N capping mod:', self.NT)

    def set_time_limit(self, timelimit):
        """Sets the specified timelimit. 
        the GA will stop if the timelimit is reached even if the primary condition is not reached.
        
        Arguments:
            timelimit {string} -- hours:minutes:seconds
        """

        timelimit = timelimit.split(':')
        hours = int(timelimit[0])
        minutes = int(timelimit[1])
        seconds = int(timelimit[2])
        self.timelimit_seconds = int(seconds + minutes * 60 + hours * 3600)
        if self.verbose:
            print('The GA will stop after', timelimit[0], 'hours,', timelimit[1], 'minutes, and', timelimit[2],
                  'seconds')

    def print_time(self):
        """print running time
        """

        hours, rem = divmod(self.time, 3600)
        minutes, seconds = divmod(rem, 60)
        print('Time {:0>2}:{:0>2}:{:0>2}'.format(int(hours), int(minutes), int(seconds)))

    def interprete(self, seq):
        """translates from 3letters code to list of numbers
        
        Arguments:
            seq {string} -- 3 letters code seq (e.g. Ala-Gly-Leu)
        
        Returns:
            string -- one letter symbol seq (e.g. AGL)
        """

        new_seq = []
        seq = seq.split('-')
        for bb in seq:
            new_seq.append(self.interprete_dict[bb])
        seq = new_seq
        return seq

    def reinterprete(self, seq):
        """translates listo fo numbers to three letters code
        
        Arguments:
            seq {string} -- one letter symbol seq (e.g. AGL)
        
        Returns:
            string -- 3 letters code seq (e.g. Ala-Gly-Leu)
        """

        new_seq = []
        for bb in seq:
            new_seq.append(self.interprete_rev_dict[bb])
        seq = '-'.join(new_seq)

        return seq

    def run(self):
        """Performs the genetic algorithm
 
        """
        startTime = time.time()

        # generation 0:
        gen = self.rndm_gen()

        if self.porpouse == 'cyclic':
            gen_cy = self.mutate_cyclic(gen)
            gen = gen_cy
        if self.verbose:
            print('Generation', self.gen_n)

        # fitness function and survival probability attribution:
        self.jd_av, self.jd_min, self.dist_dict, self.surv_dict = self.fitness_function(gen)

        if self.verbose:
            print('Average JD =', self.jd_av, 'Minimum JD =', self.jd_min)

        # progress file update (generations and their 
        # average and minimum JD from query): 
        self.write_progress()

        self.time = int(time.time() - startTime)

        if self.verbose:
            self.print_time()

        # if query is found updates found identity count (class variable):
        if self.jd_min == 0:
            self.found_identity += 1

            # updates generation number (class variable):
        self.gen_n += 1

        # default: GA runs for ten more generation after the query is found.
        while self.jd_min != 0 or self.found_identity <= 10:

            if self.timelimit_seconds is not None and self.time > self.timelimit_seconds:
                if self.verbose:
                    print('time limit reached')
                break

            if self.verbose:
                print('Generation', self.gen_n)

            # the sequences to be kept intact are chosen:
            survivors = self.who_lives()

            # n. (pop size - len(survivors)) sequences 
            # are created with crossover or mutation (cyclic):
            if self.porpouse == 'cyclic':
                new_gen = self.make_new_gen_cyclic(self.pop_size - len(survivors))
            else:
                new_gen = self.make_new_gen(self.pop_size - len(survivors))

            # the next generation is the results of merging 
            # the survivors with the new sequences:
            if self.porpouse == 'cyclic':
                gen_merg = survivors + self.mutate_cyclic(new_gen)
            else:
                gen_merg = survivors + self.mutate(new_gen)

            # eventual duplicates are removed:
            gen = remove_duplicates(gen_merg)

            if self.verbose == True:
                for s in gen[:5]:
                    print(self.reinterprete(s))

            # fitness function and survival 
            # probability attribution:
            self.jd_av, self.jd_min, self.dist_dict, self.surv_dict = self.fitness_function(gen)
            if self.verbose:
                print('Average JD =', self.jd_av, 'Minumum JD =', self.jd_min)

            # progress file update (generations and their 
            # average and minimum JD from query): 
            self.write_progress()

            self.time = int(time.time() - startTime)

            if self.verbose:
                self.print_time()

            # updates generation number (class variable):
            self.gen_n += 1

            # if query is found updates found identity count (class variable) 
            if self.jd_min == 0:
                self.found_identity += 1
