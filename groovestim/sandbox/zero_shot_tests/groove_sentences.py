hierarchy_level = 3
groove_texts_subjective = {'listen':['I do not like to listen to this music',
                          'I like to listen to this music',
                          'I love to listen to this music'],
                'dance':['I would not dance to this music',
                        'I would like to dance to this music',
                        'I love to dance to this music'],
                'party':['This music is not at all suited for a party',
                         'This music is somewhat suited for a party',
                        'This music is perfectly suited for a party'],
                'beat':['The beat of this music is hard to recognize',
                        'This music has a recognizable beat',
                        'The beat of this music is very easy to recoginze'],
                'rhythm':['This music does not have an interesting rhythm',
                         'This music has an interesting rythm',
                         'This music has a very interestin rythm'],
                'disturbing':['This music is not disturbing at all',
                            'This music is somewhat disturbing',
                                'This music is very disturbing'],
                'groove':['I would not qualify this music as groovy',
                        'I would qualify this music as somewhat groovy',
                        'I would qualify this music as very groovy']}

groove_texts_objective = {'listen':['A music that is not enjoyable to listen to',
                          'A music that is enjoyable to listen to',
                          'A music that is very enjoyable to listen to'],
                'dance':['A music that is not danceable',
                         'A music that is danceable',
                        'A music that is very danceable'],
                'party':['A music that is not at all suited for a party',
                         'A music that is somewhat suited for a party',
                        'A music that is perfectly suited for a party'],
                'beat':['A music with a beat that is hard to recognize',
                        'A music with a recognizable beat',
                        'A music with a beat that is very easy to recoginze'],
                'rhythm':['A music that does not have an interesting rhythm',
                          'A music that has an interesting rythm',
                         'A music that has a very interesting rythm'],
                'disturbing':['A music that is not disturbing',
                              'A music that is somewhat disturbing',
                              'A music that is very disturbing'],
                'groove':['A music that is not groovy',
                        'A music that is somewhat groovy',
                        'A music that is very groovy']
                              }

# ChatGPT
# Prompt which generated answers:
# We have computed data on human subjects, to evaluate their perception of groove in audio signals. Their response was obtained according to 6 questions which were :
# - "I would like to dance to this music"
# - "I like to listen to this music"
# - "This music is great for a party"
# - "The beat of this music is easy to recognize"
# - "This music has an interesting rhythm"
# - "Something in this music is disturbing"
# The 6 questions were evaluated with answers going from 1 (disagreement with the statement) to 100 (agreement with the statement).
# We would like to regroup the answers to each question according to 3 sentences: 1 sentence for the low answers, indicating disagreement, 1 sentence for the middle values, and 1 sentence for the high values, indicating agreement. Can you imagine these 3 sentences for each statement? Avoid negation as much as possible.

# Do a last one, for the notion of "groove", defined in musicology as "humans’ pleasurable urge to move their body in synchrony with music."
groove_texts_chatGPT = {'listen':["This track doesn't capture my interest for listening.",
                                            "This music is somewhat enjoyable to listen to.",
                                            "I thoroughly enjoy listening to this music."],
                                  'dance': ["This music's rhythm doesn't inspire me to dance.",
                                            "I'm somewhat inclined to dance to this music's beat.",
                                            "This music makes me want to dance immediately."],
                                  'party': ["This track doesn't fit the vibrant atmosphere of a party.",
                                            "This music could work in certain moments at a party.",
                                            "This is the perfect music to elevate a party's energy."],
                                  'beat':["Identifying the beat in this music requires effort.",
                                          "The beat of this music is noticeable with some focus.",
                                          "The beat in this music stands out clearly and compellingly."],
                                  'rhythm':["The rhythm of this music lacks captivating elements.",
                                            "There's a certain uniqueness to this music's rhythm that catches attention.",
                                            "The rhythm of this music is uniquely engaging and holds my interest."],
                                  'disturbing':["There's a harmonious quality to this music, with nothing unsettling.",
                                                "This music carries a peculiar tone that might unsettle some.",
                                                "The dissonant elements in this music create a distinctly disturbing effect."],
                                'groove':["This music doesn't evoke a desire to move in sync with its rhythm.",
                                          "This track sparks a mild urge to move along with its beat.",
                                          "The groove in this music irresistibly compels my body to move in harmony."]    
    }

def get_dictionary(dictionary_name):
        if dictionary_name == 'subjective':
                return groove_texts_subjective
        elif dictionary_name == 'objective':
                return groove_texts_objective
        elif dictionary_name == 'chatGPT':
                return groove_texts_chatGPT
        else:
                raise ValueError(f"Dictionary name not understood: {dictionary_name}")