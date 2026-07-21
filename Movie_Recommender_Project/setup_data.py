import pandas as pd

data = {
    'movie_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'title': ['The Matrix', 'Inception', 'Interstellar', 'Shutter Island', 'Prisoners', 'Avengers', 'Iron Man', 'Gladiator', 'Braveheart', 'Avatar'],
    'tags': [
        'action sci-fi hackers simulation neo',
        'action sci-fi dreams thieves mind suspense',
        'sci-fi space travel time survival',
        'suspense thriller mystery detective island',
        'suspense thriller crime kidnapping intense',
        'action superhero marvel team saving world',
        'action superhero marvel technology suit',
        'action historical rome revenge gladiator',
        'action historical scotland freedom war',
        'sci-fi action alien planet graphics'
    ],
    'rating': [8.7, 8.8, 8.6, 8.2, 8.1, 8.0, 7.9, 8.5, 8.3, 7.9],
    'year': [1999, 2010, 2014, 2010, 2013, 2012, 2008, 2000, 1995, 2009],
    'poster_url': [
        'https://images.unsplash.com/photo-1626814026160-2237a95fc5a0?w=500&h=750&fit=crop', 
        'https://images.unsplash.com/photo-1536440136628-849c177e76a1?w=500&h=750&fit=crop', 
        'https://images.unsplash.com/photo-1446776811953-b23d57bd21aa?w=500&h=750&fit=crop', 
        'https://images.unsplash.com/photo-1518104593124-ac2eeb9a4ffe?w=500&h=750&fit=crop', 
        'https://images.unsplash.com/photo-1500674425229-f692875b0ab7?w=500&h=750&fit=crop', 
        'https://images.unsplash.com/photo-1608889175123-8ee362201f81?w=500&h=750&fit=crop', 
        'https://images.unsplash.com/photo-1534447677768-be436bb09401?w=500&h=750&fit=crop', 
        'https://images.unsplash.com/photo-1473280025148-643f9b0cbac2?w=500&h=750&fit=crop', 
        'https://images.unsplash.com/photo-1599839619722-39751411ea63?w=500&h=750&fit=crop', 
        'https://images.unsplash.com/photo-1451187580459-43490279c0fa?w=500&h=750&fit=crop'  
    ]
}

df = pd.DataFrame(data)
df.to_csv('movies_dataset.csv', index=False)
print("Rich dataset 'movies_dataset.csv' created successfully!")