"""
Analogy***
The overarching system using the analogy of the roman consulship
Whereas politicians (models) compete for votes (via hyper-tuning)
to become the next consul for the next term (most influence over any new diagnosis)

A consul and senate robust enough to keep a stagnant flow of power will have the
consul elevated to a dictatorship

The consul has 50% power to choose the next plan (diagnosis)
while the other politicians (models) have the other 50% divided at first equally (50/3)
until one reaches enough influence over the others or the consul has been deemed
incompetent or lacking (for example consistently wrong diagnosis)

This is the way the roman senate selects a diagnosis of a patient
Diagnosis is****
50 % consul's choice
initial 50/3 % for each other model
Composite vote between percentages of yes and no
however if tied between consul and politicians
senate's vote is superior (cause representative democracy)!
    example rfc, k, l_regression all say 0 while XGBoost consul says 1


consul review****
new sensate
if all members are accurate: no re-arranging power

if 1 model is accurate are:
    take power from other sensate equal to
        n record of their mistakes (since new consul) times
        fraction  of their current power/total senate
    -reminder that total senate power is equal to the consul's power

if 2 models are accurate
    repeat steps from if-statement above but with power split between the 2 correct system

new consul
if consul record of mistakes is more than any senate member
if any senate member power is more than any other member and rivals consul's prestige for n cycles

methodology
1. select a random consul declare the others as senate
2. issue training/test and have roman system elect diagnosis
3. roman system issues consul review
    hyper-tuning each model
    evaluating senate and redistributing senate power
    evaluating consul and re-arranging senate i9f consul is deemed incompetent
4. Cycle and repeat until Roman consul can be elevated to dictator for life and republic can be stagnant
"""


class RomanPolitician:
    def __init__(self):
        self.alogirthim = None
        self.cycles_in_power = 0
        self.voting_power = 0
        self. max_power = 0
        self.name = str()

        # number of false positives and negatives
        self.num_mistakes = 0
        # number of True positives and negatives
        self.num_correct = 0


    def performance_score(self):
        return self.num_correct/(self.num_mistakes + self.num_correct)

    def new_reupublic(self):
        self.voting_power = 0
        # number of false positives and negatives
        self.num_mistakes = 0
        # number of True positives and negatives
        self.num_correct = 0



class RomanRepublic:
    def __init__(self, power_of_republic=100):
        self.consul = None
        self.senate_set = set()
        self.republic_power = power_of_republic / 2
        self.politican_set = set()

    # set up the roman republic system by declaring roles of each member
    def declare_republic(self, initial_consul, initial_senate_set):
        self.declare_consul(initial_consul)

        self.senate_set.clear()
        for senate_politician in initial_senate_set:
            self.declare_consul(senate_politician)

        self.politican_set = self.senate_set
        self.politican_set.add(self.consul)

    def declare_consul(self, roman_member: RomanPolitician):
        roman_member.voting_power = self.republic_power
        roman_member.max_power = self.republic_power

        roman_member.new_reupublic()
        self.consul = roman_member

    def declare_senate(self, roman_member: RomanPolitician):
        roman_member.voting_power = self.republic_power / 3
        roman_member.max_power = self.republic_power

        roman_member.new_reupublic()
        self.senate_set.add(roman_member)

    # iterate through the roman republic system given a number of cycles to train
    def cycle_through_republic(self, num_training_cycles):
        for index in range(num_training_cycles):
            self.new_republic_year()


    def new_republic_year(self):
        has_new_Consul = self.is_consul_discharged()

        if has_new_Consul:
            new_Consul = self.get_best_senator()
            self.senate_set.remove(new_Consul)
            self.senate_set.add(self.consul)

            self.declare_republic(new_Consul, self.senate_set)
        else:
            self.senate_review()

    def get_best_senator(self):
        best_senator = None

        for sentor in self.senate_set:
            if best_senator is None:
                best_senator = sentor

            if sentor.voting_power > best_senator.voting_power:
                best_senator = sentor

        return best_senator


    # re-arrange the voting power between each model in the senate
    def senate_review(self):
        pass

    # evaluate if the consul is still qualified
    def is_consul_discharged(self):
        for senator in self.senate_set:
            return ((senator.performance_score > self.consul.performance_score)
                    and (senator is self.get_best_senator()))

