package ca.behrainwala.mustafa.data;

import java.util.*;

public class TextToken {
    private final Map<String, Integer> tokens = new HashMap<>();
    private final ArrayList<String> reverseMap = new ArrayList<>();
    private final List<String> nonTokenText = Arrays.asList(",", ":", ".", "?", "-", "—", "!", "’", "`", "\"", ";", "’");
    private Integer nextToken=1;

    public static final String END="<$$>";

    private final Object lock = new Object();

    public TextToken() {
        reverseMap.add(END); // index 0 = unknown token
    }

    public void addTextToToken(String text){
        if(!nonTokenText.contains(text)){
            text = cleanText(text);
            if(!tokens.containsKey(text))
            {
                tokens.put(text, nextToken);
                reverseMap.add(text);
                synchronized (lock){
                    nextToken++;
                }
            }
        }
    }

    public int getTextToken(String text){
        text = cleanText(text);
        return tokens.get(text)==null?0:tokens.get(text);
    }

    public String getTokenText(int token){
        if(token >= 0 && token < reverseMap.size())
            return reverseMap.get(token);

        return END;
    }

    public int getTokenSize(){
        return tokens.size();
    }

    private String cleanText(String text){
        if(text==null)
            return null;

        for(String removeChar:nonTokenText) {
            text = text.replace(removeChar, "");
        }

        return text;
    }
}
