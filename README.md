
## Project Description

Fit Compass is a workout coach that helps beginners get started with exercise by providing structured workout guidance. Many beginners struggle with keeping count of reps, not knowing what exercises to do, and cheating themselves out on workouts. Fit Compass solves these problems by providing workouts for the user when logging in, counting repetitions when a movement is completed, and allowing progression only when the form is correct.

## Features
- Workout generation and workout library 
- Automated exercise repetition counting
- Anti-cheat locks and checks
- Coach avatar outfit shop
- Audio cues at 25%, 50% and 75% complete
- Email reminders at set times
- Workout summary and analysis

## Usage 

- Register in app with chosen username and password
- From homepage, read suggested workout under today's day and click start. 
- Do exercises as shown in the workout preview
- View workout stats on workout complete page or in email summary
- Navigate back to profile page
- Alternatively, choose a premade workout from the library page

## Demo
* [Presentation Slides](https://docs.google.com/presentation/d/13txgRg7zsfocPBh1YEtlXgIS29KNrTBkIIYYShhb6B8/edit?usp=sharing)
* [Demo Video](https://drive.google.com/file/d/151bJN9QF8hK1njopicsgvNtfCSkqXeif/view?usp=sharing)

## Launch prerequisites
- OpenCV 4.12
- Flask 3.1.2
- Mediapipe version 0.10.31
- Python 3.11.9
- SQLite 3.44

## Launch instructions
- clone repo
- ```bash
  python3 app.py
  ```

## Project Structure
- /templates: Templates contains the raw HTML/JS files
- /static: Static contains all the fixed media files
- static/audio: contains motivational lines
- static/costumes: contains images of full outfits with the avatar for shop preview
- static/michalis: contains the coach avatar template, seperated into arms, face and body
- static/outfits: contains images of outfits that will be overlaid on top of the templates from /michalis
- app.py: the main node of the app

## Known Issues
- consistency score not available for supermans, pushups or glute bridges
- some routes broken on history page




